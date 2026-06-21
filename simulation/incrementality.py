import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(exist_ok=True)

N_BOOTSTRAP = 200


def load_features(path: Path = PROCESSED_DIR / "criteo_features.parquet") -> pd.DataFrame:
    print(f"Loading features from {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")
    return df


def split_treatment_control(df: pd.DataFrame) -> pd.DataFrame:
    # Treatment = high click activity (ctr above median for that campaign)
    # Control = low click activity (ctr below median for that campaign)
    df = df.copy()
    median_ctr = df.groupby("campaign")["ctr"].median()
    df["median_ctr"] = df["campaign"].map(median_ctr)
    df["group"] = np.where(df["ctr"] >= df["median_ctr"], "treatment", "control")
    return df


def split_pre_post(df: pd.DataFrame) -> pd.DataFrame:
    # Pre/post split at the midpoint of the time range, per campaign
    df = df.copy()
    midpoint = df["hour_bucket"].median()
    df["period"] = np.where(df["hour_bucket"] < midpoint, "pre", "post")
    return df


def compute_did(df: pd.DataFrame) -> dict:
    grouped = df.groupby(["group", "period"])["conversion_rate"].mean().unstack()

    treatment_pre = grouped.loc["treatment", "pre"]
    treatment_post = grouped.loc["treatment", "post"]
    control_pre = grouped.loc["control", "pre"]
    control_post = grouped.loc["control", "post"]

    treatment_diff = treatment_post - treatment_pre
    control_diff = control_post - control_pre
    did_estimate = treatment_diff - control_diff

    return {
        "treatment_pre": treatment_pre,
        "treatment_post": treatment_post,
        "control_pre": control_pre,
        "control_post": control_post,
        "treatment_diff": treatment_diff,
        "control_diff": control_diff,
        "did_estimate": did_estimate,
    }


def bootstrap_confidence_interval(
    df: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []

    n = len(df)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, n)
        sample = df.iloc[sample_idx]
        try:
            result = compute_did(sample)
            estimates.append(result["did_estimate"])
        except KeyError:
            continue

    lower = np.percentile(estimates, 2.5)
    upper = np.percentile(estimates, 97.5)
    return lower, upper


def compute_incrementality_per_campaign(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    campaigns = df["campaign"].unique()
    print(f"Computing incrementality for {len(campaigns)} campaigns ...")

    for campaign in campaigns:
        campaign_df = df[df["campaign"] == campaign]

        # Need both groups and both periods present
        if campaign_df["group"].nunique() < 2 or campaign_df["period"].nunique() < 2:
            continue

        try:
            result = compute_did(campaign_df)
        except KeyError:
            continue

        rows.append({
            "campaign": campaign,
            "observed_conversion_rate": campaign_df["conversion_rate"].mean(),
            "incremental_lift": result["did_estimate"],
            "treatment_diff": result["treatment_diff"],
            "control_diff": result["control_diff"],
        })

    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    df = load_features()
    df = split_treatment_control(df)
    df = split_pre_post(df)

    print("\nOverall DiD estimate (across all campaigns pooled):")
    overall = compute_did(df)
    for k, v in overall.items():
        print(f"  {k}: {v:.6f}")

    print("\nComputing bootstrap confidence interval (this may take a moment) ...")
    lower, upper = bootstrap_confidence_interval(df)
    print(f"  95% CI: [{lower:.6f}, {upper:.6f}]")

    per_campaign = compute_incrementality_per_campaign(df)
    per_campaign = per_campaign.sort_values("incremental_lift", ascending=False)

    print(f"\nPer-campaign incrementality (top 10 by lift):")
    print(per_campaign.head(10).to_string(index=False))

    print(f"\nPer-campaign incrementality (bottom 10 by lift):")
    print(per_campaign.tail(10).to_string(index=False))

    path = RESULTS_DIR / "incrementality_results.parquet"
    per_campaign.to_parquet(path, index=False)
    print(f"\nSaved to {path}")

    return per_campaign


if __name__ == "__main__":
    run()