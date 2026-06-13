import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from prefect import results
from prefect import results

from forecast.model import (
    load_features,
    train_test_split_temporal,
    load_model,
    predict,
    QUANTILES,
    FEATURE_COLS,
    TARGET_COL,
)

PLOTS_DIR = Path("forecast/plots")
PLOTS_DIR.mkdir(exist_ok=True)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def evaluate_all_quantiles(test: pd.DataFrame) -> dict:
    results = {}
    y_true = test[TARGET_COL].values

    for q in QUANTILES:
        model = load_model(q)
        y_pred = predict(model, test)
        loss = pinball_loss(y_true, y_pred, q)
        results[q] = {"y_pred": y_pred, "loss": loss}
        print(f"Quantile {q} — Pinball Loss: {loss:.6f}")

    return results


def plot_calibration(test: pd.DataFrame, results: dict) -> None:
    y_true = test[TARGET_COL].values

    # For each quantile, what fraction of actuals fall below the prediction?
    # A well-calibrated q10 model should have ~10% of actuals below it
    print("\nCalibration check (actual coverage vs expected):")
    for q in QUANTILES:
        y_pred = results[q]["y_pred"]
        coverage = float(np.mean(y_true <= y_pred))
        print(f"  q{int(q*100)}: expected {q:.0%} coverage, actual {coverage:.1%}")


def plot_predictions(test: pd.DataFrame, results: dict, n_campaigns: int = 3) -> None:
    campaigns = test["campaign"].unique()[:n_campaigns]

    fig, axes = plt.subplots(n_campaigns, 1, figsize=(12, 4 * n_campaigns))
    if n_campaigns == 1:
        axes = [axes]

    for ax, campaign in zip(axes, campaigns):
        subset = test[test["campaign"] == campaign].sort_values("hour_bucket")
        x = subset["hour_bucket"].values
        y_true = subset[TARGET_COL].values

        subset_positions = subset.index - test.index[0]
        q10 = results[0.1]["y_pred"][subset_positions]
        q50 = results[0.5]["y_pred"][subset_positions]
        q90 = results[0.9]["y_pred"][subset_positions]

        ax.fill_between(x, q10, q90, alpha=0.3, label="80% interval (q10-q90)")
        ax.plot(x, q50, label="q50 (median)", linewidth=1.5)
        ax.plot(x, y_true, label="Actual", linewidth=1, linestyle="--")
        ax.set_title(f"Campaign {campaign}")
        ax.set_xlabel("Hour bucket")
        ax.set_ylabel("Conversion rate")
        ax.legend()

    plt.tight_layout()
    path = PLOTS_DIR / "calibration_plot.png"
    plt.savefig(path)
    print(f"\nPlot saved to {path}")


def run() -> None:
    df = load_features()
    _, test = train_test_split_temporal(df)

    print(f"Evaluating on {len(test):,} test rows ...\n")
    results = evaluate_all_quantiles(test)
    plot_calibration(test, results)
    plot_predictions(test, results)


if __name__ == "__main__":
    run()