import pandas as pd
import numpy as np
from pathlib import Path
import pyomo.environ as pyo

SCENARIOS_DIR = Path("data/scenarios")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(exist_ok=True)

TOTAL_BUDGET = 1000.0
MAX_CAMPAIGN_BUDGET = 200.0
CVAR_ALPHA = 0.2  # Focus on worst 20% of scenarios
LAMBDA = 0.5      # Balance between average and CVaR


def load_scenarios(path: Path = SCENARIOS_DIR / "scenarios.parquet") -> pd.DataFrame:
    print(f"Loading scenarios from {path} ...")
    df = pd.read_parquet(path)
    df["conversion_rate"] = df["conversion_rate"].clip(lower=0.0)
    print(f"Loaded {len(df):,} rows — {df['scenario'].nunique()} scenarios, {df['campaign'].nunique()} campaigns")
    return df


def build_scenario_dict(scenarios: pd.DataFrame) -> dict:
    scenario_dict = {}
    for _, row in scenarios.iterrows():
        scenario_dict[(row["campaign"], row["scenario"])] = row["conversion_rate"]
    return scenario_dict


def build_model(
    scenarios: pd.DataFrame,
    total_budget: float,
    max_campaign_budget: float,
    cvar_alpha: float = CVAR_ALPHA,
    lambda_weight: float = LAMBDA,
) -> pyo.ConcreteModel:

    campaigns = list(scenarios["campaign"].unique())
    scenario_ids = list(scenarios["scenario"].unique())
    n_scenarios = len(scenario_ids)
    n_tail = max(1, int(np.floor(cvar_alpha * n_scenarios)))
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
    model.n_tail = pyo.Param(initialize=n_tail)
    model.lambda_weight = pyo.Param(initialize=lambda_weight)

    # Stage 1 — budget allocation
    model.budget = pyo.Var(
        model.campaigns,
        domain=pyo.NonNegativeReals,
        bounds=(0, max_campaign_budget)
    )

    # Stage 2 — conversions per campaign per scenario
    model.conversions = pyo.Var(
        model.campaigns,
        model.scenarios,
        domain=pyo.NonNegativeReals
    )

    # Total conversions per scenario (sum across campaigns)
    model.total_conversions_scenario = pyo.Var(
        model.scenarios,
        domain=pyo.NonNegativeReals
    )

    # CVaR auxiliary variables
    # eta = VaR threshold (the cutoff between tail and non-tail scenarios)
    model.eta = pyo.Var(domain=pyo.Reals)

    # u_s = shortfall below eta in scenario s (how much worse than eta)
    model.u = pyo.Var(
        model.scenarios,
        domain=pyo.NonNegativeReals
    )

    # Objective — weighted combination of average and CVaR
    avg_conversions = (1.0 / n_scenarios) * sum(
        model.total_conversions_scenario[s]
        for s in model.scenarios
    )

    cvar = model.eta - (1.0 / n_tail) * sum(
        model.u[s] for s in model.scenarios
    )

    model.objective = pyo.Objective(
        expr=(1 - lambda_weight) * avg_conversions + lambda_weight * cvar,
        sense=pyo.maximize
    )

    # Budget constraint
    model.budget_constraint = pyo.Constraint(
        expr=sum(model.budget[c] for c in model.campaigns) <= model.total_budget
    )

    # Conversion linking constraint
    def conversion_rule(model, c, s):
        return model.conversions[c, s] == model.conversion_rate[c, s] * model.budget[c]

    model.conversion_constraint = pyo.Constraint(
        model.campaigns,
        model.scenarios,
        rule=conversion_rule
    )

    # Total conversions per scenario
    def total_conversion_rule(model, s):
        return model.total_conversions_scenario[s] == sum(
            model.conversions[c, s] for c in model.campaigns
        )

    model.total_conversion_constraint = pyo.Constraint(
        model.scenarios,
        rule=total_conversion_rule
    )

    # CVaR shortfall constraints
    # u_s >= eta - total_conversions_s (shortfall below the VaR threshold)
    def shortfall_rule(model, s):
        return model.u[s] >= model.eta - model.total_conversions_scenario[s]

    model.shortfall_constraint = pyo.Constraint(
        model.scenarios,
        rule=shortfall_rule
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


def extract_results(model: pyo.ConcreteModel) -> tuple[pd.DataFrame, dict]:
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

    scenario_totals = [
        pyo.value(model.total_conversions_scenario[s])
        for s in model.scenarios
    ]
    scenario_totals_sorted = sorted(scenario_totals)
    n_tail = pyo.value(model.n_tail)
    cvar_value = np.mean(scenario_totals_sorted[:int(n_tail)])

    metrics = {
        "total_spend": df["budget_allocated"].sum(),
        "avg_conversions": df["avg_expected_conversions"].sum(),
        "cvar": cvar_value,
        "worst_scenario": min(scenario_totals),
        "best_scenario": max(scenario_totals),
        "eta": pyo.value(model.eta),
    }

    return df, metrics


def run(total_budget: float = TOTAL_BUDGET) -> tuple[pd.DataFrame, dict]:
    scenarios = load_scenarios()

    print(f"Building CVaR stochastic LP (lambda={LAMBDA}, alpha={CVAR_ALPHA}) ...")
    model = build_model(scenarios, total_budget, MAX_CAMPAIGN_BUDGET)
    model = solve(model)

    results, metrics = extract_results(model)

    print(f"\nTotal budget:      {total_budget:.2f}")
    print(f"Total spend:       {metrics['total_spend']:.2f}")
    print(f"Avg conversions:   {metrics['avg_conversions']:.4f}")
    print(f"CVaR (worst 20%):  {metrics['cvar']:.4f}")
    print(f"Worst scenario:    {metrics['worst_scenario']:.4f}")
    print(f"Best scenario:     {metrics['best_scenario']:.4f}")
    print(f"Campaigns funded:  {(results['budget_allocated'] > 0.01).sum()}")
    print(f"\nTop 10 campaigns by allocation:")
    print(results.head(10).to_string(index=False))

    path = RESULTS_DIR / "cvar_results.parquet"
    results.to_parquet(path, index=False)
    print(f"\nSaved to {path}")

    return results, metrics


if __name__ == "__main__":
    run()