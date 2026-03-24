# 🏦 Predictive Modeling & Risk Scoring for Bank Customer Churn

> A machine learning-driven churn intelligence system for European retail banking — featuring real-time risk scoring, explainable predictions, and an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Results](#models--results)
- [Feature Engineering](#feature-engineering)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Deliverables](#deliverables)
- [Tech Stack](#tech-stack)

---

## Overview

Customer churn is one of the most costly challenges in retail banking. This project builds a full predictive churn intelligence pipeline that:

- Trains and evaluates **4 machine learning models** on 10,000 European bank customers
- Generates **calibrated churn probability scores** (0–1) per customer
- Identifies and ranks **key behavioral and financial churn drivers**
- Delivers an **interactive Streamlit dashboard** for real-time risk assessment and what-if scenario analysis

The overall churn rate in the dataset is **20.37%**, with Germany exhibiting the highest regional churn at **32.4%**.

---

## Dataset

**Source:** European Central Bank — Retail Analytics Program (`European_Bank.csv`)

| Column | Type | Description |
|---|---|---|
| `CustomerId` | int | Unique customer identifier (dropped) |
| `Surname` | str | Customer surname (dropped) |
| `CreditScore` | int | Creditworthiness score (300–900) |
| `Geography` | str | France / Germany / Spain |
| `Gender` | str | Male / Female |
| `Age` | int | Customer age |
| `Tenure` | int | Years with the bank (0–10) |
| `Balance` | float | Account balance (€) |
| `NumOfProducts` | int | Number of bank products held (1–4) |
| `HasCrCard` | int | Credit card ownership (0/1) |
| `IsActiveMember` | int | Active member status (0/1) |
| `EstimatedSalary` | float | Annual estimated salary (€) |
| `Exited` | int | **Target** — churned (1) or retained (0) |

---

## Project Structure

```
bank-churn-prediction/
│
├── data/
│   └── European_Bank.csv          # Raw dataset
│
├── outputs/
│   ├── model_comparison.png       # Bar chart: all model metrics
│   ├── roc_curves.png             # ROC curves for all models
│   ├── feature_importance.png     # Top 15 RF feature importances
│   ├── confusion_matrices.png     # Confusion matrices (RF + GB)
│   ├── probability_distribution.png  # Churn score distribution
│   └── eda_demographics.png       # EDA: geography, gender, age
│
├── model_artifacts/
│   ├── best_model.pkl             # Serialized Random Forest model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── feature_cols.pkl           # Ordered feature column list
│   ├── feature_importance.csv     # Feature importance scores
│   └── metrics.json               # All model evaluation metrics
│
├── streamlit_app.py               # Interactive dashboard (5 tabs)
├── churn_model.py                 # Training pipeline + plot generation
├── Churn_Research_Paper.docx      # Full academic research paper
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-org/bank-churn-prediction.git
cd bank-churn-prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
# Copy European_Bank.csv into the data/ directory
```

### `requirements.txt`

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
joblib>=1.3.0
```

---

## Usage

### Train Models & Generate Plots

Runs the full preprocessing, feature engineering, model training, evaluation, and plot export pipeline:

```bash
python churn_model.py
```

Outputs saved to `outputs/` and `model_artifacts/`.

### Launch the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501` by default.

---

## Models & Results

All models were trained with **stratified 80/20 split** and **class-balanced weights** to handle the 20.37% churn minority class.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 70.9% | 38.3% | 70.3% | 0.496 | 0.776 |
| Decision Tree | 75.8% | 44.5% | 76.9% | 0.564 | 0.826 |
| **Random Forest** ⭐ | **84.6%** | **61.3%** | **66.1%** | **0.636** | **0.865** |
| Gradient Boosting | 86.7% | 77.8% | 48.2% | 0.595 | 0.863 |

**Recommended model: Random Forest** — highest F1 and ROC-AUC, best balance between precision and recall for retention campaigns.

---

## Feature Engineering

Six derived features were added beyond the raw dataset:

| Feature | Formula | Purpose |
|---|---|---|
| `Balance_Salary_Ratio` | Balance / (Salary + 1) | Financial dependency on the bank |
| `Product_Density` | NumOfProducts / (Tenure + 1) | Product acquisition rate |
| `Engagement_Product` | IsActiveMember × NumOfProducts | Combined depth of engagement |
| `Age_Tenure_Interaction` | Age × Tenure | Long-term loyalty signal |
| `Zero_Balance` | 1 if Balance = 0 | Dormant/inactive account flag |
| `Senior_Customer` | 1 if Age > 50 | Elevated churn risk segment |

---

## Streamlit Dashboard

The dashboard is organized into five navigation tabs:

| Tab | Description |
|---|---|
| 📊 **Executive Dashboard** | KPIs, churn by geography, age group, activity status, and product count |
| 🤖 **Model Performance** | Metrics table, ROC curves, confusion matrix, feature importance slider |
| 🔍 **Risk Calculator** | Input a customer profile → get real-time churn probability + risk tier |
| 🔄 **What-If Simulator** | Compare two profiles side-by-side and observe delta risk |
| 📈 **EDA Insights** | Balance/credit distributions, correlation heatmap, tenure trend, raw data |

Risk tiers in the calculator:

- 🟢 **Low Risk** — probability < 30%: standard engagement
- 🟡 **Medium Risk** — 30–60%: targeted campaign
- 🔴 **High Risk** — > 60%: immediate retention action

---

## Deliverables

| File | Description |
|---|---|
| `streamlit_app.py` | Full interactive Streamlit dashboard |
| `churn_model.py` | Model training and evaluation pipeline |
| `Churn_Research_Paper.docx` | 8-section academic research paper |
| `outputs/*.png` | Six publication-ready visualization charts |
| `model_artifacts/` | Serialized models, scaler, and metrics |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Report Generation | docx.js (Node.js) |

---

## Key Insights

- **Germany** has the highest churn rate at 32.4% — nearly double France and Spain
- Customers with **exactly 2 products** have the lowest churn; those with 3–4 products churn at extreme rates (83–100%)
- **Inactive members** churn at 26.9% vs. 14.3% for active members
- **Age 40–60** is the highest-risk demographic segment
- **Female customers** churn at 25.1% vs. 16.5% for male customers

---

*European Central Bank — Retail Analytics Division | Unified Mentor Research Program*
