"""
predictor.py — Future Expense Prediction using Linear Regression
=================================================================

ALGORITHM: Linear Regression (scikit-learn)
--------------------------------------------
How it works:
  1. Load monthly expense totals from CSV (grouped by month)
  2. Encode months as integers: [0, 1, 2, ... n-1]  (X)
  3. Use monthly totals as target values             (y)
  4. Fit a Linear Regression: y = slope * X + intercept
  5. Predict the next 3 months by passing [n, n+1, n+2]

The model learns the overall spending trend (going up, down, or flat)
and extrapolates it into the future.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
from dateutil.relativedelta import relativedelta


def load_monthly_totals(filepath: str) -> pd.DataFrame:
    """
    Load CSV and aggregate expenses into monthly totals.

    Args:
        filepath: path to data.csv

    Returns:
        DataFrame with columns [month_label, month_index, total]
        sorted chronologically
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M")

    monthly = (
        df.groupby("Month")["Amount"]
        .sum()
        .reset_index()
        .sort_values("Month")
    )
    monthly["month_index"] = range(len(monthly))
    monthly["month_label"] = monthly["Month"].dt.strftime("%b %Y")
    monthly.rename(columns={"Amount": "total"}, inplace=True)

    return monthly[["month_label", "month_index", "total", "Month"]]


def train_model(monthly_df: pd.DataFrame):
    """
    Fit a Linear Regression model on monthly spending data.

    Args:
        monthly_df: output of load_monthly_totals()

    Returns:
        tuple: (fitted LinearRegression model, metrics dict)
    """
    X = monthly_df[["month_index"]].values   # shape (n, 1)
    y = monthly_df["total"].values           # shape (n,)

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = {
        "slope":       round(model.coef_[0], 2),
        "intercept":   round(model.intercept_, 2),
        "r2_score":    round(r2_score(y, y_pred), 4),
        "mae":         round(mean_absolute_error(y, y_pred), 2),
        "n_months":    len(monthly_df),
    }
    return model, metrics


def predict_future(
    model: LinearRegression,
    monthly_df: pd.DataFrame,
    months_ahead: int = 3
) -> list[dict]:
    """
    Predict expenses for the next N months.

    Args:
        model:        fitted LinearRegression model
        monthly_df:   historical monthly data
        months_ahead: how many future months to forecast

    Returns:
        list of dicts: [{month_label, predicted_amount, month_index}]
    """
    last_index = monthly_df["month_index"].max()
    last_month = monthly_df["Month"].max()

    predictions = []
    for i in range(1, months_ahead + 1):
        future_index  = last_index + i
        future_period = last_month + i
        future_label  = future_period.strftime("%b %Y")
        predicted_amt = max(0.0, model.predict([[future_index]])[0])
        predictions.append({
            "month_label":      future_label,
            "month_index":      future_index,
            "predicted_amount": round(predicted_amt, 2),
        })
    return predictions


def run_prediction(filepath: str, months_ahead: int = 3) -> dict:
    """
    Full prediction pipeline: load → train → predict.

    Args:
        filepath:     path to data.csv
        months_ahead: number of future months to forecast

    Returns:
        dict with keys:
          'historical'  → list of {month_label, total} dicts
          'predictions' → list of {month_label, predicted_amount} dicts
          'metrics'     → model performance metrics
          'trend'       → 'increasing' | 'decreasing' | 'stable'
    """
    monthly_df = load_monthly_totals(filepath)

    if len(monthly_df) < 2:
        return {
            "error": "Need at least 2 months of data to train the model.",
            "n_months": len(monthly_df)
        }

    model, metrics = train_model(monthly_df)
    predictions    = predict_future(model, monthly_df, months_ahead)

    slope = metrics["slope"]
    if slope > 100:
        trend = "increasing"
    elif slope < -100:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "historical":  monthly_df[["month_label", "total"]].to_dict("records"),
        "predictions": predictions,
        "metrics":     metrics,
        "trend":       trend,
    }


def format_results(result: dict) -> str:
    """Pretty-print prediction results."""
    if "error" in result:
        return f"Error: {result['error']}"

    m = result["metrics"]
    lines = [
        "=" * 52,
        "  EXPENSE PREDICTOR — Linear Regression Results",
        "=" * 52,
        f"  Months of data : {m['n_months']}",
        f"  Slope          : ₹{m['slope']:,.2f} / month",
        f"  Intercept      : ₹{m['intercept']:,.2f}",
        f"  R² score       : {m['r2_score']} (1.0 = perfect fit)",
        f"  MAE            : ₹{m['mae']:,.2f}",
        f"  Trend          : {result['trend'].upper()}",
        "-" * 52,
        "  Historical spending:",
    ]
    for h in result["historical"]:
        lines.append(f"    {h['month_label']:<12} ₹{h['total']:>10,.2f}")

    lines += ["-" * 52, "  Forecasted spending:"]
    for p in result["predictions"]:
        lines.append(f"    {p['month_label']:<12} ₹{p['predicted_amount']:>10,.2f}  ← predicted")

    lines.append("=" * 52)
    return "\n".join(lines)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate CSV data for testing without a real file
    import io

    sample_csv = """Date,Category,Description,Amount
2024-10-05,Food,Groceries,3200
2024-10-12,Transport,Uber,800
2024-10-18,Entertainment,Netflix,500
2024-10-25,Housing,Rent,9000
2024-11-03,Food,Groceries,3500
2024-11-10,Transport,Bus pass,600
2024-11-20,Housing,Rent,9000
2024-11-28,Entertainment,Movie,700
2024-12-02,Food,Groceries,4100
2024-12-15,Transport,Ola,950
2024-12-22,Housing,Rent,9000
2024-12-28,Health,Pharmacy,400
2025-01-04,Food,Groceries,3800
2025-01-11,Transport,Metro,550
2025-01-19,Housing,Rent,9000
2025-01-25,Entertainment,Zomato,1200
"""
    # Write sample to a temp file and run
    with open("/tmp/sample_data.csv", "w") as f:
        f.write(sample_csv)

    print("\nRunning Linear Regression predictor...")
    result = run_prediction("/tmp/sample_data.csv", months_ahead=3)
    print(format_results(result))
