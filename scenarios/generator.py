import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube

PROCESSED_DIR = Path("data/processed")
SCENARIOS_DIR = Path("data/scenarios")
SCENARIOS_DIR.mkdir(exist_ok=True)

N_SCENARIOS = 20
QUANTILES = [0.1, 0.5, 0.9]


def load_predictions(path: Path = PROCESSED_DIR / "quantile_predictions.parquet") -> pd.DataFrame:
    print(f"Loading quantile predictions from {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")
    return df


def fit_lognormal(q10: float, q50: float, q90: float) -> tuple[float, float]:
    # q50 is the median of a lognormal → mu = log(q50)
    # q10 and q90 give us sigma via the normal quantile function
    # norm.ppf(0.9) = 1.2816 — the z-score at the 90th percentile
    epsilon = 1e-8
    q10 = max(q10, epsilon)
    q50 = max(q50, epsilon)
    q90 = max(q90, epsilon)

    mu = np.log(q50)
    sigma = (np.log(q90) - np.log(q10)) / (2 * norm.ppf(0.9))
    sigma = max(sigma, epsilon)
    return mu, sigma


def generate_scenarios(
    predictions: pd.DataFrame, n_scenarios: int = N_SCENARIOS
) -> pd.DataFrame:
    campaigns = predictions["campaign"].unique()
    print(f"Generating {n_scenarios} scenarios for {len(campaigns)} campaigns ...")

    # Latin Hypercube sampler — one sample per scenario
    sampler = LatinHypercube(d=1)
    # Generate n_scenarios samples in [0, 1] — these are probabilities
    lhs_samples = sampler.random(n=n_scenarios).flatten()

    scenario_rows = []

    for campaign in campaigns:
        row = predictions[predictions["campaign"] == campaign].iloc[0]
        q10 = row["q10"]
        q50 = row["q50"]
        q90 = row["q90"]

        mu, sigma = fit_lognormal(q10, q50, q90)

        # Convert LHS probability samples to actual conversion rate values
        # norm.ppf converts a probability to a z-score
        # then we apply the lognormal inverse CDF: exp(mu + sigma * z)
        z_scores = norm.ppf(lhs_samples)
        scenario_values = np.clip(np.exp(mu + sigma * z_scores), 0.0, 1.0)

        for i, value in enumerate(scenario_values):
            scenario_rows.append({
                "campaign": campaign,
                "scenario": i,
                "conversion_rate": float(value),
            })

    df = pd.DataFrame(scenario_rows)
    print(f"Generated {len(df):,} scenario rows ({len(campaigns)} campaigns × {n_scenarios} scenarios)")
    return df


def save_scenarios(df: pd.DataFrame) -> None:
    path = SCENARIOS_DIR / "scenarios.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved to {path}")


def generate_predictions_from_models(features: pd.DataFrame) -> pd.DataFrame:
    import pickle
    from forecast.model import FEATURE_COLS

    MODEL_DIR = Path("forecast/models")
    quantile_map = {0.1: "q10", 0.5: "q50", 0.9: "q90"}

    # Use the last available hour per campaign as the forecast input
    latest = (
        features.sort_values("hour_bucket")
        .groupby("campaign")
        .last()
        .reset_index()
    )

    for q, col in quantile_map.items():
        model_path = MODEL_DIR / f"lgb_q{int(q * 100)}.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        latest[col] = model.predict(latest[FEATURE_COLS])

    return latest[["campaign", "q10", "q50", "q90"]]


def run() -> pd.DataFrame:
    from forecast.model import load_features

    features = load_features()
    predictions = generate_predictions_from_models(features)

    # Save predictions for reference
    pred_path = PROCESSED_DIR / "quantile_predictions.parquet"
    predictions.to_parquet(pred_path, index=False)
    print(f"Saved quantile predictions to {pred_path}")
    print(predictions.describe())

    scenarios = generate_scenarios(predictions)
    save_scenarios(scenarios)
    return scenarios


if __name__ == "__main__":
    run()