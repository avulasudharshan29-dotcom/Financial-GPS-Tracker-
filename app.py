"""
app.py — Financial GPS: Main Streamlit Application
====================================================
Integrates:
  - Expense tracker  (pandas + data.csv)
  - Category analysis (matplotlib / st.bar_chart)
  - Budget optimizer  (optimizer.py → Generate-and-Test)
  - Expense predictor (predictor.py → Linear Regression)
  - Financial advice  (rule-based system)

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import date, datetime

from optimizer import run_generate_and_test
from predictor import load_monthly_totals, train_model, predict_future

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH   = "data.csv"
CATEGORIES = ["Food", "Transport", "Housing", "Health", "Entertainment", "Other"]
CAT_COLORS = {
    "Food": "#1d9e75", "Transport": "#3266ad", "Housing": "#ba7517",
    "Health": "#a32d2d", "Entertainment": "#7f77dd", "Other": "#888780",
}

st.set_page_config(
    page_title="Financial GPS",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-box{background:#f8f7f4;border-radius:10px;padding:1rem 1.2rem;border:1px solid #e8e6df}
  .section-title{font-size:13px;font-weight:600;color:#6b6b67;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.75rem}
  .advice-ok{background:#e1f5ee;border-left:3px solid #1d9e75;padding:.6rem .9rem;border-radius:0 6px 6px 0;margin:.4rem 0;font-size:14px}
  .advice-warn{background:#faeeda;border-left:3px solid #ba7517;padding:.6rem .9rem;border-radius:0 6px 6px 0;margin:.4rem 0;font-size:14px}
  .advice-bad{background:#fcebeb;border-left:3px solid #a32d2d;padding:.6rem .9rem;border-radius:0 6px 6px 0;margin:.4rem 0;font-size:14px}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    cols = ["Date", "Category", "Description", "Amount"]
    return pd.DataFrame(columns=cols)


def save_data(df: pd.DataFrame):
    df.to_csv(CSV_PATH, index=False)


def get_category_totals(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return df.groupby("Category")["Amount"].sum().to_dict()


def fmt(n: float) -> str:
    return f"₹{n:,.0f}"


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Financial GPS")
    st.markdown("---")
    monthly_budget = st.number_input(
        "Monthly Budget (₹)", min_value=1000, max_value=500000,
        value=25000, step=500
    )
    st.markdown("---")
    tab = st.radio(
        "Navigate",
        ["📋 Tracker", "📊 Analysis", "⚙️ Optimizer", "🔮 Predictor", "💡 Advice"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Data stored in `data.csv`")


# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
total_spent    = df["Amount"].sum() if not df.empty else 0
budget_left    = monthly_budget - total_spent
cat_totals     = get_category_totals(df)
top_cat        = max(cat_totals, key=cat_totals.get) if cat_totals else "—"


# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Spent",    fmt(total_spent))
c2.metric("Budget Left",    fmt(budget_left),    delta=f"{budget_left/monthly_budget*100:.0f}% remaining")
c3.metric("Top Category",   top_cat,             delta=fmt(cat_totals.get(top_cat, 0)) if top_cat != "—" else None)
c4.metric("Entries Logged", len(df))

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# TAB: TRACKER
# ════════════════════════════════════════════════════════════════════════════
if tab == "📋 Tracker":
    st.subheader("Expense Tracker")
    col_form, col_table = st.columns([1, 2])

    with col_form:
        st.markdown('<div class="section-title">Add Expense</div>', unsafe_allow_html=True)
        with st.form("add_expense_form", clear_on_submit=True):
            exp_date  = st.date_input("Date", value=date.today())
            exp_cat   = st.selectbox("Category", CATEGORIES)
            exp_desc  = st.text_input("Description", placeholder="e.g. Grocery run")
            exp_amt   = st.number_input("Amount (₹)", min_value=0.0, step=50.0)
            submitted = st.form_submit_button("➕ Add Expense", use_container_width=True)

        if submitted:
            if exp_amt <= 0:
                st.error("Please enter a valid amount.")
            else:
                new_row = pd.DataFrame([{
                    "Date":        pd.Timestamp(exp_date),
                    "Category":    exp_cat,
                    "Description": exp_desc or exp_cat,
                    "Amount":      exp_amt,
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                save_data(df)
                st.success(f"Added {fmt(exp_amt)} for {exp_cat}")
                st.rerun()

        # Budget progress bar
        st.markdown('<div class="section-title" style="margin-top:1.5rem">Budget Usage</div>', unsafe_allow_html=True)
        pct = min(1.0, total_spent / monthly_budget)
        color = "normal" if pct < 0.7 else ("off" if pct < 0.9 else "inverse")
        st.progress(pct)
        st.caption(f"{pct*100:.1f}% of {fmt(monthly_budget)} used")

        st.download_button(
            "⬇️ Download data.csv",
            data=df.to_csv(index=False).encode(),
            file_name="data.csv",
            mime="text/csv",
        )

    with col_table:
        st.markdown('<div class="section-title">Expense Log</div>', unsafe_allow_html=True)
        if df.empty:
            st.info("No expenses yet. Add one using the form.")
        else:
            display_df = df.sort_values("Date", ascending=False).copy()
            display_df["Date"]   = display_df["Date"].dt.strftime("%d %b %Y")
            display_df["Amount"] = display_df["Amount"].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(display_df[["Date", "Category", "Description", "Amount"]],
                         use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB: ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif tab == "📊 Analysis":
    st.subheader("Spending Analysis")

    if df.empty:
        st.info("Add some expenses to see analysis.")
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-title">Spending by Category</div>', unsafe_allow_html=True)
            cat_df = pd.DataFrame(
                list(cat_totals.items()), columns=["Category", "Amount"]
            ).sort_values("Amount", ascending=False)
            st.bar_chart(cat_df.set_index("Category"), color="#3266ad")

        with col_r:
            st.markdown('<div class="section-title">Category Share (%)</div>', unsafe_allow_html=True)
            total = sum(cat_totals.values()) or 1
            pie_df = pd.DataFrame([
                {"Category": k, "Share": round(v / total * 100, 1)}
                for k, v in sorted(cat_totals.items(), key=lambda x: -x[1])
            ])
            st.dataframe(pie_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title" style="margin-top:1.5rem">Monthly Trend</div>', unsafe_allow_html=True)
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
        trend_df = df.groupby("Month")["Amount"].sum().reset_index()
        st.bar_chart(trend_df.set_index("Month"), color="#1d9e75")


# ════════════════════════════════════════════════════════════════════════════
# TAB: OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════
elif tab == "⚙️ Optimizer":
    st.subheader("Budget Optimizer — Generate-and-Test Algorithm")

    st.markdown("""
    **How the algorithm works:**
    1. 🎲 **Generate** — randomly create a budget allocation for each category, 
       respecting minimum essential spend floors
    2. 🧪 **Test** — score the candidate: how close is the total to your savings target?
    3. 🔁 **Repeat** — run N trials and keep the best-scoring allocation
    4. ✅ **Return** — output the allocation that best meets your goal
    """)

    col_cfg, col_res = st.columns([1, 2])

    with col_cfg:
        savings_goal = st.number_input("Savings goal (₹/month)", min_value=0, value=5000, step=500)
        n_trials     = st.slider("Search trials", min_value=100, max_value=5000, value=1000, step=100)
        run_btn      = st.button("🚀 Run Optimizer", use_container_width=True, type="primary")

    if run_btn:
        if not cat_totals:
            st.error("No expense data found. Add some expenses first.")
        else:
            with st.spinner(f"Running {n_trials} Generate-and-Test iterations..."):
                result = run_generate_and_test(
                    category_totals=cat_totals,
                    savings_goal=savings_goal,
                    n_trials=n_trials
                )

            with col_res:
                st.success(f"✅ Projected savings: {fmt(result['projected_savings'])}/month")
                st.caption(f"Search deviation: {fmt(result['score'])} from target | Trials: {result['trials']}")

                rows = []
                for cat in sorted(result["actual_totals"], key=lambda x: -result["actual_totals"][x]):
                    cur = result["actual_totals"][cat]
                    rec = result["allocation"][cat]
                    cut = cur - rec
                    rows.append({
                        "Category":    cat,
                        "Current (₹)": f"{cur:,.0f}",
                        "Recommended (₹)": f"{rec:,.0f}",
                        "Save (₹)":    f"{cut:,.0f}",
                        "Cut %":       f"{cut/cur*100:.1f}%" if cur > 0 else "0%",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB: PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
elif tab == "🔮 Predictor":
    st.subheader("Future Expense Predictor — Linear Regression")

    st.markdown("""
    **How the model works:**
    - Monthly totals are encoded as integers `[0, 1, 2 … n]` (X-axis)
    - Linear Regression fits: **y = slope × month + intercept**
    - The model predicts the next 3 months by extrapolating the trend line
    """)

    if df.empty:
        st.info("Add expense data across multiple months to enable predictions.")
    else:
        monthly_df = load_monthly_totals(CSV_PATH)

        if len(monthly_df) < 2:
            st.warning("Need at least 2 months of data. Keep tracking!")
        else:
            model, metrics = train_model(monthly_df)
            predictions    = predict_future(model, monthly_df, months_ahead=3)

            col_m, col_p = st.columns(2)

            with col_m:
                st.markdown('<div class="section-title">Model Metrics</div>', unsafe_allow_html=True)
                st.metric("Slope (₹/month)", f"₹{metrics['slope']:,.0f}",
                          delta="spending rising" if metrics['slope'] > 0 else "spending falling")
                st.metric("R² Score",  f"{metrics['r2_score']:.3f}", delta="fit quality (1.0 = perfect)")
                st.metric("MAE",       f"₹{metrics['mae']:,.0f}", delta="mean absolute error")

            with col_p:
                st.markdown('<div class="section-title">3-Month Forecast</div>', unsafe_allow_html=True)
                for p in predictions:
                    st.metric(p["month_label"], fmt(p["predicted_amount"]))

            st.markdown('<div class="section-title" style="margin-top:1rem">Trend Chart</div>', unsafe_allow_html=True)
            hist = monthly_df[["month_label", "total"]].rename(columns={"total": "Actual"})
            pred_df = pd.DataFrame(predictions).rename(columns={"predicted_amount": "Predicted"})[["month_label", "Predicted"]]
            combined = pd.merge(hist, pred_df, on="month_label", how="outer").set_index("month_label")
            st.line_chart(combined, color=["#3266ad", "#1d9e75"])


# ════════════════════════════════════════════════════════════════════════════
# TAB: ADVICE
# ════════════════════════════════════════════════════════════════════════════
elif tab == "💡 Advice":
    st.subheader("Financial Advice — Rule-Based System")

    st.markdown("""
    The rule engine evaluates your spending data against a set of 
    expert-defined financial rules and generates personalised recommendations.
    """)

    if df.empty or not cat_totals:
        st.info("Add expense data to receive personalised financial advice.")
    else:
        total = sum(cat_totals.values()) or 1
        pct   = lambda cat: (cat_totals.get(cat, 0) / total * 100)
        savings_rate = (monthly_budget - total_spent) / monthly_budget * 100

        rules = []

        # Rule 1: Overall budget
        if total_spent > monthly_budget:
            rules.append(("bad",  f"⚠️ Over Budget: You've spent {fmt(total_spent)}, exceeding your {fmt(monthly_budget)} budget by {fmt(total_spent - monthly_budget)}."))
        elif total_spent > monthly_budget * 0.85:
            rules.append(("warn", f"↑ Approaching Limit: {total_spent/monthly_budget*100:.0f}% of budget used. Watch discretionary spending."))
        else:
            rules.append(("ok",   f"✓ On Track: Only {total_spent/monthly_budget*100:.0f}% of budget used."))

        # Rule 2: Food spending
        if pct("Food") > 35:
            rules.append(("warn", f"↓ High Food Spend: {pct('Food'):.0f}% of total. Try meal planning or cooking at home to reduce costs."))
        elif pct("Food") > 0:
            rules.append(("ok",   f"✓ Food spend is healthy at {pct('Food'):.0f}% of total."))

        # Rule 3: Entertainment cap
        if pct("Entertainment") > 20:
            rules.append(("warn", f"↓ Entertainment at {pct('Entertainment'):.0f}%. Consider a monthly cap of {fmt(monthly_budget * 0.12)}."))
        elif pct("Entertainment") > 0:
            rules.append(("ok",   f"✓ Entertainment spend ({pct('Entertainment'):.0f}%) is within healthy limits."))

        # Rule 4: Transport costs
        if pct("Transport") > 25:
            rules.append(("warn", f"↓ High Transport Cost: {pct('Transport'):.0f}% of spending. Explore public transit or carpooling."))

        # Rule 5: Health awareness
        if cat_totals.get("Health", 0) == 0:
            rules.append(("warn", "↓ No Health Spending detected. Ensure you're not skipping checkups or medications."))
        else:
            rules.append(("ok",   f"✓ Health spending recorded: {fmt(cat_totals['Health'])}."))

        # Rule 6: Savings rate
        if savings_rate > 20:
            rules.append(("ok",  f"✓ Great savings rate: {savings_rate:.0f}% of budget saved. Consider SIP or emergency fund."))
        elif savings_rate > 0:
            rules.append(("warn", f"↑ Savings rate: {savings_rate:.0f}%. Aim for at least 20% monthly savings."))
        else:
            rules.append(("bad",  "⚠️ Negative savings this month. Urgent review of discretionary expenses recommended."))

        # Rule 7: Housing ratio
        if pct("Housing") > 50:
            rules.append(("warn", f"↓ Housing at {pct('Housing'):.0f}% of spend — above the recommended 30-40%. Review housing costs."))

        for rtype, msg in rules:
            st.markdown(f'<div class="advice_{rtype}">{msg}</div>', unsafe_allow_html=True)
