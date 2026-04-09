# 🧭 Financial GPS

> An AI-powered personal finance tracker with budget optimization, expense prediction, and smart financial advice — built with Python and Streamlit.

---

## 📌 Project Overview

Financial GPS is a locally-run personal finance application that helps you track, analyze, and optimize your monthly spending. It combines classical AI techniques with machine learning to give actionable financial insights.

| Module | File | Algorithm |
|--------|------|-----------|
| Expense Tracker | `app.py` | pandas + CSV |
| Spending Analysis | `app.py` | pandas groupby / charts |
| Budget Optimizer | `optimizer.py` | Generate-and-Test Search |
| Expense Predictor | `predictor.py` | Linear Regression (scikit-learn) |
| Financial Advice | `app.py` | Rule-Based System |

---

## 🗂️ Project Structure

```
financial-gps/
├── app.py           → Main Streamlit UI (all 5 tabs)
├── optimizer.py     → Generate-and-Test budget optimizer
├── predictor.py     → Linear Regression expense predictor
├── data.csv         → Expense data store (auto-created)
└── README.md        → This file
```

---

## ⚙️ Setup & Installation

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/financial-gps.git
cd financial-gps
```

### 2. Install dependencies

```bash
pip install streamlit pandas scikit-learn python-dateutil
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 🚀 Features

### 📋 Tab 1 — Expense Tracker
- Add expenses with date, category, description, and amount
- All data stored locally in `data.csv` using pandas
- Live budget usage progress bar
- Export your data as a CSV file anytime

**Categories supported:** Food, Transport, Housing, Health, Entertainment, Other

---

### 📊 Tab 2 — Spending Analysis
- Category-wise bar chart showing where money goes
- Percentage share table per category
- Monthly trend chart across all recorded months
- Automatically updates as you add expenses

---

### ⚙️ Tab 3 — Budget Optimizer (Generate-and-Test)

**Algorithm: Generate-and-Test Search**

This is a stochastic AI search algorithm that works in three steps:

```
Step 1 — GENERATE
  For each category, randomly pick a budget value between:
    floor = category_minimum  (e.g. Housing: 75% of current spend)
    ceiling = current actual spend

Step 2 — TEST
  Score the candidate:
    score = |proposed_total − savings_target|
  Lower score = better fit

Step 3 — REPEAT
  Run N trials (default: 1000), keep the best candidate
```

**Category spending floors** (minimum allowed cuts):

| Category | Floor |
|----------|-------|
| Housing | 75% — mostly fixed costs |
| Health | 80% — essential |
| Food | 60% — some flexibility |
| Transport | 45% — carpool / transit options |
| Other | 35% — general discretionary |
| Entertainment | 20% — most cuttable |

**How to use:**
1. Set your monthly savings goal (₹)
2. Adjust the number of search trials (100–5000)
3. Click **Run Optimizer**
4. Review the recommended allocation per category

---

### 🔮 Tab 4 — Expense Predictor (Linear Regression)

**Algorithm: Linear Regression**

```
X = [0, 1, 2, ... n]          ← month index
y = [total_jan, total_feb, ...]← monthly expense totals

Model: y = slope × X + intercept
Predicts: months n+1, n+2, n+3
```

Requires at least **2 months of data** to generate predictions.

**Metrics shown:**
- **Slope** — how fast spending is rising/falling per month
- **R² Score** — model fit quality (1.0 = perfect)
- **MAE** — mean absolute error in ₹

---

### 💡 Tab 5 — Financial Advice (Rule-Based System)

A set of expert-defined financial rules evaluated against your data:

| Rule | Condition | Type |
|------|-----------|------|
| Budget check | Spent > budget | Danger |
| Budget warning | Spent > 85% of budget | Warning |
| Food spending | Food > 35% of total | Warning |
| Entertainment cap | Entertainment > 20% of total | Warning |
| Transport costs | Transport > 25% of total | Warning |
| Health awareness | No health entries | Warning |
| Savings rate | Savings > 20% of budget | Good |
| Housing ratio | Housing > 50% of total | Warning |

---

## 📄 data.csv Format

The CSV is created automatically when you add your first expense. Format:

```
Date,Category,Description,Amount
2025-04-01,Food,Grocery shopping,3200
2025-04-05,Transport,Uber,850
2025-04-10,Housing,Rent,9000
```

You can also manually edit `data.csv` or import your own file in this format.

---

## 🧪 Running Modules Standalone

Test the optimizer independently:

```bash
python optimizer.py
```

Test the predictor independently:

```bash
python predictor.py
```

Both scripts include built-in sample data for quick testing.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥ 1.30 | Web UI framework |
| pandas | ≥ 2.0 | Data storage and analysis |
| scikit-learn | ≥ 1.3 | Linear Regression model |
| python-dateutil | ≥ 2.8 | Date arithmetic for predictions |
| numpy | ≥ 1.24 | Numerical operations |

---

## 🛠️ Troubleshooting

**`ModuleNotFoundError`** — Run `pip install streamlit pandas scikit-learn python-dateutil`

**Predictor shows "Need at least 2 months"** — Add expenses across different calendar months (e.g. March and April entries)

**`data.csv` not found** — Add your first expense through the Tracker tab and it will be created automatically

**Port already in use** — Run `streamlit run app.py --server.port 8502`

---

## 📝 License

This project is for educational purposes. Feel free to use, modify, and extend it.

---

*Built with Python · pandas · scikit-learn · Streamlit*
