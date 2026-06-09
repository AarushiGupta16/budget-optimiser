import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/criteo_attribution_dataset.tsv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

COLUMNS_NEEDED = ["timestamp", "campaign", "conversion", "click", "cost"]

SENTINEL_COLUMNS = [
    "conversion_timestamp",
    "click_pos",
    "click_nb",
    "time_since_last_click",
    "conversion_id",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    print(f"Loading raw data from {path} ...")
    df = pd.read_csv(path, sep="\t")
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def validate(df: pd.DataFrame) -> None:
    assert len(df) > 0, "Dataframe is empty"
    expected_columns = [
        "timestamp", "uid", "campaign", "conversion",
        "conversion_timestamp", "conversion_id", "attribution",
        "click", "click_pos", "click_nb", "cost", "cpo",
        "time_since_last_click", "cat1", "cat2", "cat3",
        "cat4", "cat5", "cat6", "cat7", "cat8", "cat9"
    ]
    missing = set(expected_columns) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    print("Validation passed")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Replace sentinel -1 values with NaN
    df[SENTINEL_COLUMNS] = df[SENTINEL_COLUMNS].replace(-1, np.nan)

    # Keep only columns needed for the optimiser
    df = df[COLUMNS_NEEDED].copy()

    # Fix types
    df["timestamp"] = df["timestamp"].astype(int)
    df["campaign"] = df["campaign"].astype(str)
    df["conversion"] = df["conversion"].astype(int)
    df["click"] = df["click"].astype(int)
    df["cost"] = df["cost"].astype(float)

    print(f"Cleaned dataframe: {df.shape}")
    print(df.dtypes)
    return df


def save(df: pd.DataFrame, path: Path = PROCESSED_DIR / "criteo_clean.parquet") -> None:
    df.to_parquet(path, index=False)
    print(f"Saved to {path}")


def run() -> pd.DataFrame:
    df = load_raw()
    validate(df)
    df = clean(df)
    save(df)
    return df


if __name__ == "__main__":
    run()