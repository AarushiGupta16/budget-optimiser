import pandas as pd
import numpy as np
from pathlib import Path
import pyomo.environ as pyo

SCENARIOS_DIR = Path("data/scenarios")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(exist_ok=True)

TOTAL_BUDGET = 1000.0
MAX_CAMPAIGN_BUDGET = 200.0


def load_scenarios(path: Path = SCENARIOS_DIR / "scenarios.parquet") -> pd.DataFrame:
    print(f"Loading scenarios from {path} ...")
    df = pd.read_parquet(path)
    df["conversion_rate"] = df["conversion_rate"].clip(lower=0.0)
    print(f"Loaded {len(df):,} rows — {df['scenario'].nunique()} scenarios, {df['campaign'].nunique()} campaigns")
    return df


def build_scenario_dict(scenarios: pd.DataFrame) -> dict:
    # Build a nested dict: {(campaign, scenario): conversion_rate}
    scenario_dict = {}
    for _, row in scenarios.iterrows():
        scenario_dict[(row["campaign"], row["scenario"])] = row["conversion_rate"]
    return scenario_dict


def build_model(
    scenarios: pd.DataFrame,
    total_budget: float,
    max_campaign_budget: float,
) -> pyo.ConcreteModel:

    campaigns = list(scenarios["campaign"].unique())
    scenario_ids = list(scenarios["scenario"].unique())
    n_scenarios = len(scenario_ids)
    scenario_dict = build_scenario_dict(scenarios)

    model = pyo.ConcreteModel()

    # Sets
    model.campaigns = pyo.Set(initialize=campaigns)
    model.scenarios = pyo.Set(initialize=scenario_ids)

    # Parameters
    model.conversion_rate = pyo.Param(
        model.campaigns,
        model.scenarios,
        initialize=scenario_dict
    )
    model.total_budget = pyo.Param(initialize=total_budget)
    model.max_budget = pyo.Param(initialize=max_campaign_budget)
    model.n_scenarios = pyo.Param(initialize=n_scenarios)

    # Stage 1 decision variable — budget allocation (same across all scenarios)
    model.budget = pyo.Var(
        model.campaigns,
        domain=pyo.NonNegativeReals,
        bounds=(0, max_campaign_budget)
    )

    # Stage 2 variable — conversions received in each scenario
    model.conversions = pyo.Var(
        model.campaigns,
        model.scenarios,
        domain=pyo.NonNegativeReals
    )

    # Objective — maximise average conversions across all scenarios
    model.objective = pyo.Objective(
        expr=(1.0 / n_scenarios) * sum(
            model.conversions[c, s]
            for c in model.campaigns
            for s in model.scenarios
        ),
        sense=pyo.maximize
    )

    # Budget constraint — total spend cannot exceed budget
    model.budget_constraint = pyo.Constraint(
        expr=sum(model.budget[c] for c in model.campaigns) <= model.total_budget
    )

    # Conversion constraint — conversions = conversion_rate × budget for each scenario
    def conversion_rule(model, c, s):
        return model.conversions[c, s] == model.conversion_rate[c, s] * model.budget[c]

    model.conversion_constraint = pyo.Constraint(
        model.campaigns,
        model.scenarios,
        rule=conversion_rule
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
        budget = pyo.value(model.budget[c])
        avg_conversions = np.mean([
            pyo.value(model.conversions[c, s])
            for s in model.scenarios
        ])
        rows.append({
            "campaign": c,
            "budget_allocated": budget,
            "avg_expected_conversions": avg_conversions,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("budget_allocated", ascending=False).reset_index(drop=True)
    return df


def run(total_budget: float = TOTAL_BUDGET) -> pd.DataFrame:
    scenarios = load_scenarios()

    print(f"Building two-stage stochastic LP ...")
    model = build_model(scenarios, total_budget, MAX_CAMPAIGN_BUDGET)
    model = solve(model)

    results = extract_results(model)
    total_conversions = results["avg_expected_conversions"].sum()
    total_spend = results["budget_allocated"].sum()
    campaigns_funded = (results["budget_allocated"] > 0).sum()

    print(f"\nTotal budget: {total_budget:.2f}")
    print(f"Total spend:  {total_spend:.2f}")
    print(f"Campaigns funded: {campaigns_funded}")
    print(f"Expected conversions (avg across scenarios): {total_conversions:.4f}")
    print(f"\nTop 10 campaigns by allocation:")
    print(results.head(10).to_string(index=False))

    path = RESULTS_DIR / "stochastic_results.parquet"
    results.to_parquet(path, index=False)
    print(f"\nSaved to {path}")

    return results


if __name__ == "__main__":
    run()