import pandas as pd
import numpy as np
from pathlib import Path
import pyomo.environ as pyo

SCENARIOS_DIR = Path("data/scenarios")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(exist_ok=True)

TOTAL_BUDGET = 1000.0
MAX_CAMPAIGN_BUDGET = 200.0


def load_predictions(path: Path = PROCESSED_DIR / "quantile_predictions.parquet") -> pd.DataFrame:
    print(f"Loading quantile predictions from {path} ...")
    df = pd.read_parquet(path)
    return df


def build_model(predictions: pd.DataFrame, total_budget: float, max_campaign_budget: float) -> pyo.ConcreteModel:
    campaigns = predictions["campaign"].tolist()
    conversion_rates = dict(zip(predictions["campaign"], predictions["q50"]))

    # Replace negative predictions with zero
    conversion_rates = {c: max(v, 0.0) for c, v in conversion_rates.items()}

    model = pyo.ConcreteModel()

    # Sets
    model.campaigns = pyo.Set(initialize=campaigns)

    # Parameters
    model.conversion_rate = pyo.Param(
        model.campaigns,
        initialize=conversion_rates
    )
    model.total_budget = pyo.Param(initialize=total_budget)
    model.max_budget = pyo.Param(initialize=max_campaign_budget)

    # Decision variables — how much budget to allocate to each campaign
    model.budget = pyo.Var(
        model.campaigns,
        domain=pyo.NonNegativeReals,
        bounds=(0, max_campaign_budget)
    )

    # Objective — maximise total expected conversions
    model.objective = pyo.Objective(
        expr=sum(
            model.conversion_rate[c] * model.budget[c]
            for c in model.campaigns
        ),
        sense=pyo.maximize
    )

    # Constraint — total spend cannot exceed budget
    model.budget_constraint = pyo.Constraint(
        expr=sum(model.budget[c] for c in model.campaigns) <= model.total_budget
    )

    return model


def solve(model: pyo.ConcreteModel) -> pyo.ConcreteModel:
    solver = pyo.SolverFactory("glpk")
    result = solver.solve(model, tee=False)

    status = result.solver.termination_condition
    print(f"Solver status: {status}")

    if status != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solver did not find optimal solution: {status}")

    return model


def extract_results(model: pyo.ConcreteModel) -> pd.DataFrame:
    rows = []
    for c in model.campaigns:
        rows.append({
            "campaign": c,
            "budget_allocated": pyo.value(model.budget[c]),
            "conversion_rate": pyo.value(model.conversion_rate[c]),
            "expected_conversions": pyo.value(model.conversion_rate[c]) * pyo.value(model.budget[c]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("budget_allocated", ascending=False).reset_index(drop=True)
    return df


def run(total_budget: float = TOTAL_BUDGET) -> pd.DataFrame:
    predictions = load_predictions()
    print(f"Building deterministic LP for {len(predictions)} campaigns ...")

    model = build_model(predictions, total_budget, MAX_CAMPAIGN_BUDGET)
    model = solve(model)

    results = extract_results(model)
    total_conversions = results["expected_conversions"].sum()
    total_spend = results["budget_allocated"].sum()

    print(f"\nTotal budget: {total_budget:.2f}")
    print(f"Total spend:  {total_spend:.2f}")
    print(f"Expected conversions: {total_conversions:.4f}")
    print(f"\nTop 10 campaigns by allocation:")
    print(results.head(10).to_string(index=False))

    path = RESULTS_DIR / "deterministic_results.parquet"
    results.to_parquet(path, index=False)
    print(f"\nSaved to {path}")

    return results


if __name__ == "__main__":
    run()