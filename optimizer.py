"""
optimizer.py — Budget Optimization using Generate-and-Test Search Algorithm
============================================================================

ALGORITHM: Generate-and-Test (a classic AI search strategy)
------------------------------------------------------------
How it works:
  1. GENERATE  → randomly create a candidate budget allocation for each category
                 within allowed floor (minimum spend) and ceiling (current spend)
  2. TEST      → score the candidate: how close is its total to the savings target?
  3. REPEAT    → run N trials and keep track of the best-scoring candidate
  4. RETURN    → the allocation with the lowest score (best fit)

This is a stochastic search — it explores the solution space randomly,
making it simple, explainable, and effective for small budget problems.
"""

import random
import pandas as pd


# ── Category spending floors (minimum fraction of current spend to preserve) ──
# Essential categories have higher floors (can't cut too deep)
CATEGORY_FLOORS = {
    "Food":          0.60,   # must keep at least 60% of food budget
    "Housing":       0.75,   # rent/utilities are mostly fixed
    "Health":        0.80,   # health is non-negotiable
    "Transport":     0.45,   # some flexibility (WFH, carpooling)
    "Entertainment": 0.20,   # most cuttable category
    "Other":         0.35,   # general discretionary
}
DEFAULT_FLOOR = 0.40  # fallback for unknown categories


def generate_candidate(category_totals: dict) -> dict:
    """
    GENERATE step: randomly allocate budget for each category.

    For each category, pick a random value between:
      - floor  = minimum allowed spend (fraction of current)
      - ceiling = current actual spend (can't exceed what was spent)

    Args:
        category_totals: dict of {category: actual_spent}

    Returns:
        dict of {category: proposed_budget}
    """
    candidate = {}
    for category, actual in category_totals.items():
        floor_frac = CATEGORY_FLOORS.get(category, DEFAULT_FLOOR)
        floor_amt  = actual * floor_frac
        ceiling    = actual  # never suggest spending MORE than current
        candidate[category] = random.uniform(floor_amt, ceiling)
    return candidate


def test_candidate(candidate: dict, target_total: float) -> float:
    """
    TEST step: score a candidate allocation.

    Score = absolute difference between candidate total and target total.
    Lower score = better fit (closer to the savings goal).

    Args:
        candidate:    dict of {category: proposed_budget}
        target_total: the total spend we're aiming for (current - savings_goal)

    Returns:
        float score (lower is better)
    """
    proposed_total = sum(candidate.values())
    return abs(proposed_total - target_total)


def run_generate_and_test(
    category_totals: dict,
    savings_goal: float,
    n_trials: int = 1000,
    seed: int = 42
) -> dict:
    """
    Main Generate-and-Test search loop.

    Runs n_trials iterations of:
      1. Generate a random candidate allocation
      2. Test (score) it against the target
      3. Keep it if it's the best seen so far

    Args:
        category_totals: dict of {category: actual_monthly_spend}
        savings_goal:    how much to save per month (₹)
        n_trials:        number of random candidates to generate
        seed:            random seed for reproducibility

    Returns:
        dict with keys:
          'allocation'     → {category: recommended_budget}
          'actual_totals'  → original spending per category
          'current_total'  → total current spend
          'target_total'   → target spend after savings
          'projected_savings' → how much will actually be saved
          'score'          → how close we got to the target
          'trials'         → number of trials run
    """
    random.seed(seed)

    current_total = sum(category_totals.values())
    target_total  = max(0, current_total - savings_goal)

    if not category_totals:
        return {"error": "No expense data provided."}

    best_candidate = None
    best_score     = float("inf")

    for _ in range(n_trials):
        # ── GENERATE ──
        candidate = generate_candidate(category_totals)

        # ── TEST ──
        score = test_candidate(candidate, target_total)

        # ── KEEP BEST ──
        if score < best_score:
            best_score     = score
            best_candidate = candidate

    projected_savings = current_total - sum(best_candidate.values())

    return {
        "allocation":         {k: round(v, 2) for k, v in best_candidate.items()},
        "actual_totals":      {k: round(v, 2) for k, v in category_totals.items()},
        "current_total":      round(current_total, 2),
        "target_total":       round(target_total, 2),
        "projected_savings":  round(projected_savings, 2),
        "score":              round(best_score, 2),
        "trials":             n_trials,
    }


def optimize_from_csv(filepath: str, savings_goal: float, n_trials: int = 1000) -> dict:
    """
    Load expense data from CSV and run the optimizer.

    CSV format: Date, Category, Description, Amount

    Args:
        filepath:     path to data.csv
        savings_goal: monthly savings target in ₹
        n_trials:     number of Generate-and-Test iterations

    Returns:
        optimization result dict (see run_generate_and_test)
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Use only the most recent month's data for optimization
    df["Date"] = pd.to_datetime(df["Date"])
    latest_month = df["Date"].dt.to_period("M").max()
    df_month = df[df["Date"].dt.to_period("M") == latest_month]

    category_totals = (
        df_month.groupby("Category")["Amount"]
        .sum()
        .to_dict()
    )

    return run_generate_and_test(category_totals, savings_goal, n_trials)


def format_results(result: dict) -> str:
    """Pretty-print the optimization results."""
    if "error" in result:
        return f"Error: {result['error']}"

    lines = [
        "=" * 52,
        "  BUDGET OPTIMIZER — Generate-and-Test Results",
        "=" * 52,
        f"  Current total spend : ₹{result['current_total']:,.2f}",
        f"  Target total spend  : ₹{result['target_total']:,.2f}",
        f"  Projected savings   : ₹{result['projected_savings']:,.2f}",
        f"  Search score        : ₹{result['score']:,.2f} (deviation)",
        f"  Trials run          : {result['trials']}",
        "-" * 52,
        f"  {'Category':<18} {'Current':>10} {'Recommended':>12} {'Cut':>8}",
        "-" * 52,
    ]
    for cat in result["actual_totals"]:
        current = result["actual_totals"][cat]
        rec     = result["allocation"][cat]
        cut     = current - rec
        lines.append(
            f"  {cat:<18} ₹{current:>8,.0f}   ₹{rec:>9,.0f}  -₹{cut:>5,.0f}"
        )
    lines.append("=" * 52)
    return "\n".join(lines)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_totals = {
        "Food":          8500,
        "Housing":       9000,
        "Transport":     3200,
        "Entertainment": 2800,
        "Health":        1500,
        "Other":         1200,
    }

    print("\nRunning Generate-and-Test optimizer...")
    result = run_generate_and_test(
        category_totals=sample_totals,
        savings_goal=5000,
        n_trials=1000
    )
    print(format_results(result))
