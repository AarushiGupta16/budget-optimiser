import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
INPUT_PATH = PROCESSED_DIR / "criteo_clean.parquet"
OUTPUT_PATH = PROCESSED_DIR / "criteo_features.parquet"


def load_clean(path: Path = INPUT_PATH) -> pd.DataFrame:
    print(f"Loading cleaned data from {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")
    return df


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    # Each timestamp unit is one second
    # Convert to hour bucket: integer divide by 3600
    df["hour_bucket"] = df["timestamp"] // 3600

    agg = (
        df.groupby(["campaign", "hour_bucket"])
        .agg(
            impressions=("conversion", "count"),
            conversions=("conversion", "sum"),
            clicks=("click", "sum"),
            spend=("cost", "sum"),
        )
        .reset_index()
    )

    print(f"Aggregated to {len(agg):,} hourly campaign rows")
    return agg


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # Avoid division by zero with np.where
    df["ctr"] = np.where(
        df["impressions"] > 0,
        df["clicks"] / df["impressions"],
        0.0
    )

    df["conversion_rate"] = np.where(
        df["impressions"] > 0,
        df["conversions"] / df["impressions"],
        0.0
    )

    df["cpc"] = np.where(
        df["clicks"] > 0,
        df["spend"] / df["clicks"],
        0.0
    )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"] = df["hour_bucket"] % 24
    df["day_of_week"] = (df["hour_bucket"] // 24) % 7
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["campaign", "hour_bucket"]).copy()

    for window in [3, 24]:
        df[f"conversion_rate_roll{window}"] = (
            df.groupby("campaign")["conversion_rate"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"spend_roll{window}"] = (
            df.groupby("campaign")["spend"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    return df


def save(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    df.to_parquet(path, index=False)
    print(f"Saved to {path}")


def run() -> pd.DataFrame:
    df = load_clean()
    df = aggregate_hourly(df)
    df = add_derived_features(df)
    df = add_time_features(df)
    df = add_rolling_features(df)
    print(f"Final feature set: {df.shape}")
    print(df.dtypes)
    save(df)
    return df


if __name__ == "__main__":
    run()