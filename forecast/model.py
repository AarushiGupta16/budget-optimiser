import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import pickle

PROCESSED_DIR = Path("data/processed")
INPUT_PATH = PROCESSED_DIR / "criteo_features.parquet"
MODEL_DIR = Path("forecast/models")
MODEL_DIR.mkdir(exist_ok=True)

QUANTILES = [0.1, 0.5, 0.9]

FEATURE_COLS = [
    "impressions",
    "clicks",
    "spend",
    "ctr",
    "cpc",
    "hour_of_day",
    "day_of_week",
    "conversion_rate_roll3",
    "spend_roll3",
    "conversion_rate_roll24",
    "spend_roll24",
]

TARGET_COL = "conversion_rate"


def load_features(path: Path = INPUT_PATH) -> pd.DataFrame:
    print(f"Loading features from {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")
    return df


def train_test_split_temporal(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Sort by time and split — never shuffle time series data
    df = df.sort_values("hour_bucket").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


def train_quantile_model(
    train: pd.DataFrame, quantile: float
) -> lgb.Booster:
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]

    params = {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "verbosity": -1,
    }

    dataset = lgb.Dataset(X_train, label=y_train)
    print(f"Training quantile={quantile} model ...")
    model = lgb.train(params, dataset)
    return model


def save_model(model: lgb.Booster, quantile: float) -> None:
    path = MODEL_DIR / f"lgb_q{int(quantile * 100)}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {path}")


def load_model(quantile: float) -> lgb.Booster:
    path = MODEL_DIR / f"lgb_q{int(quantile * 100)}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def predict(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[FEATURE_COLS])


def run() -> dict:
    df = load_features()
    train, test = train_test_split_temporal(df)

    models = {}
    for q in QUANTILES:
        model = train_quantile_model(train, q)
        save_model(model, q)
        models[q] = model

    # Quick sanity check on test set
    sample = test.head(5)[FEATURE_COLS]
    print("\nSample predictions on test set (first 5 rows):")
    for q in QUANTILES:
        preds = predict(models[q], sample)
        print(f"  q{int(q*100)}: {preds.round(4)}")

    return models


if __name__ == "__main__":
    run()