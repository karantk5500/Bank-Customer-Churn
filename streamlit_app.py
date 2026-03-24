#!/usr/bin/env python3
"""
Bank Customer Churn Prediction Dashboard
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0f0f1a; }
  .block-container { padding: 1.5rem 2rem; }
  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #7209b7;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #c77dff; }
  .metric-label { font-size: 0.75rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
  .risk-high   { background: linear-gradient(135deg, #3b0a0a, #7b0000); border:1px solid #ff4444; border-radius:12px; padding:1rem; }
  .risk-medium { background: linear-gradient(135deg, #3b2000, #7b4a00); border:1px solid #ff9900; border-radius:12px; padding:1rem; }
  .risk-low    { background: linear-gradient(135deg, #0a2a0a, #004d00); border:1px solid #00cc44; border-radius:12px; padding:1rem; }
  h1, h2, h3 { color: #e0aaff !important; }
  .stTabs [data-baseweb="tab"] { color: #c77dff; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Data & Model Loading ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('European_Bank.csv')
    return df

@st.cache_resource
def train_models(df):
    data = df.copy()
    data.drop(columns=['CustomerId', 'Surname', 'Year'], inplace=True, errors='ignore')
    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=False)

    # Feature Engineering
    data['Balance_Salary_Ratio']   = data['Balance'] / (data['EstimatedSalary'] + 1)
    data['Product_Density']        = data['NumOfProducts'] / (data['Tenure'] + 1)
    data['Engagement_Product']     = data['IsActiveMember'] * data['NumOfProducts']
    data['Age_Tenure_Interaction'] = data['Age'] * data['Tenure']
    data['Zero_Balance']           = (data['Balance'] == 0).astype(int)
    data['Senior_Customer']        = (data['Age'] > 50).astype(int)

    feature_cols = [c for c in data.columns if c != 'Exited']
    X = data[feature_cols]; y = data['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), True),
        'Decision Tree':       (DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced'), False),
        'Random Forest':       (RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1), False),
        'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42), False),
    }
    trained, results = {}, {}
    for name, (model, use_scaled) in models.items():
        Xt = X_train_sc if use_scaled else X_train
        Xv = X_test_sc  if use_scaled else X_test
        model.fit(Xt, y_train)
        yp = model.predict(Xv)
        ypr = model.predict_proba(Xv)[:, 1]
        trained[name] = model
        results[name] = dict(
            Accuracy=accuracy_score(y_test, yp),
            Precision=precision_score(y_test, yp),
            Recall=recall_score(y_test, yp),
            F1=f1_score(y_test, yp),
            AUC=roc_auc_score(y_test, ypr),
            y_pred=yp, y_proba=ypr
        )

    feat_imp = pd.Series(
        trained['Random Forest'].feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return trained, results, scaler, feature_cols, X_test, y_test, feat_imp

df_raw = load_data()

with st.spinner("Training models on 10,000 customer records..."):
    models, results, scaler, feature_cols, X_test, y_test, feat_imp = train_models(df_raw)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏦 Churn Intelligence")
st.sidebar.markdown("*European Central Bank – Retail Analytics*")
st.sidebar.divider()

nav = st.sidebar.radio("Navigation", [
    "📊 Executive Dashboard",
    "🤖 Model Performance",
    "🔍 Risk Calculator",
    "🔄 What-If Simulator",
    "📈 EDA Insights",
])

st.sidebar.divider()
st.sidebar.metric("Dataset Size", "10,000 customers")
st.sidebar.metric("Churn Rate", f"{df_raw['Exited'].mean():.1%}")
st.sidebar.metric("Best Model F1", f"{results['Random Forest']['F1']:.3f}")

# ─── TAB 1: Executive Dashboard ───────────────────────────────────────────────
if nav == "📊 Executive Dashboard":
    st.title("📊 Executive Churn Intelligence Dashboard")
    st.markdown("*Predictive Risk Scoring for European Retail Banking Customers*")

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Customers",   f"{len(df_raw):,}",          "👥"),
        ("Churned",           f"{df_raw['Exited'].sum():,}", "⚠️"),
        ("Churn Rate",        f"{df_raw['Exited'].mean():.1%}", "📉"),
        ("RF Accuracy",       f"{results['Random Forest']['Accuracy']:.1%}", "🎯"),
        ("RF ROC-AUC",        f"{results['Random Forest']['AUC']:.3f}", "📈"),
    ]
    for col, (label, val, icon) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.5rem">{icon}</div>
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([1,1])

    # Churn by geography
    with col_l:
        st.subheader("🗺️ Churn Rate by Geography")
        geo = df_raw.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
        geo_values = np.asarray(geo.values, dtype=float)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#0f0f1a')
        ax.set_facecolor('#0f0f1a')
        bars = ax.bar(geo.index, geo_values, color=['#c77dff','#7209b7','#480ca8'])
        ax.set_ylabel('Churn Rate', color='white'); ax.set_title('', color='white')
        for bar, v in zip(bars, geo_values):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.004, f'{v:.1%}', ha='center', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
        ax.set_ylim(0, 0.35)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Churn by num products
    with col_r:
        st.subheader("📦 Churn Rate by Number of Products")
        prod = df_raw.groupby('NumOfProducts')['Exited'].mean()
        prod_values = np.asarray(prod.values, dtype=float)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
        ax.bar(prod.index.astype(str), prod_values, color=['#4cc9f0','#7209b7','#f72585','#f72585'])
        ax.set_xlabel('Products', color='white'); ax.set_ylabel('Churn Rate', color='white')
        for i,(idx,v) in enumerate(prod.items()):
            ax.text(i, v+0.01, f'{v:.1%}', ha='center', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    col_l2, col_r2 = st.columns([1,1])
    with col_l2:
        st.subheader("👤 Churn by Age Group")
        df_raw['AgeGroup'] = pd.cut(df_raw['Age'], bins=[0,30,40,50,60,100], labels=['<30','30-40','40-50','50-60','60+'])
        ag = df_raw.groupby('AgeGroup', observed=True)['Exited'].mean()
        ag_values = np.asarray(ag.values, dtype=float)
        fig, ax = plt.subplots(figsize=(5,3.5))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
        ax.bar(ag.index, ag_values, color='#e0aaff')
        ax.set_ylabel('Churn Rate', color='white')
        for i,v in enumerate(ag_values):
            ax.text(i, v+0.005, f'{v:.1%}', ha='center', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_r2:
        st.subheader("🏅 Activity vs Churn")
        act = df_raw.groupby('IsActiveMember')['Exited'].mean()
        act_values = np.asarray(act.values, dtype=float)
        fig, ax = plt.subplots(figsize=(5,3.5))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
        ax.bar(['Inactive','Active'], act_values, color=['#f72585','#4cc9f0'])
        ax.set_ylabel('Churn Rate', color='white')
        for i,v in enumerate(act_values):
            ax.text(i, v+0.005, f'{v:.1%}', ha='center', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ─── TAB 2: Model Performance ─────────────────────────────────────────────────
elif nav == "🤖 Model Performance":
    st.title("🤖 Model Performance Analysis")

    # Metrics Table
    st.subheader("📋 Comparative Metrics")
    metrics_df = pd.DataFrame({
        name: {k: round(v,4) for k,v in r.items() if isinstance(v, float)}
        for name, r in results.items()
    }).T
    metrics_df.columns = ['Accuracy','Precision','Recall','F1','ROC-AUC']
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='#7209b7').format("{:.4f}"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 ROC Curves")
        palette = {'Logistic Regression':'#4361ee','Decision Tree':'#f72585',
                   'Random Forest':'#c77dff','Gradient Boosting':'#4cc9f0'}
        fig, ax = plt.subplots(figsize=(6,5))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
        for name in results:
            fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
            ax.plot(fpr, tpr, label=f"{name} ({results[name]['AUC']:.3f})", color=palette[name], lw=2)
        ax.plot([0,1],[0,1],'--', color='#555', lw=1)
        ax.set_xlabel('FPR', color='white'); ax.set_ylabel('TPR', color='white')
        ax.set_title('ROC Curves', color='white')
        ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("🔲 Confusion Matrix (Random Forest)")
        fig, ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
        cm = confusion_matrix(y_test, results['Random Forest']['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                    xticklabels=['Retained','Churned'],
                    yticklabels=['Retained','Churned'], cbar=False,
                    annot_kws={'size':14, 'weight':'bold'})
        ax.set_xlabel('Predicted', color='white'); ax.set_ylabel('Actual', color='white')
        ax.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.subheader("🌟 Feature Importance (Random Forest)")
    top_n = st.slider("Show top N features", 5, len(feat_imp), 15)
    top = feat_imp.head(top_n)
    colors = plt.cm.get_cmap('plasma')(np.linspace(0.3, 0.9, top_n))
    fig, ax = plt.subplots(figsize=(9, top_n * 0.45 + 1))
    fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel('Importance', color='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
    for i,(idx,v) in enumerate(zip(top.index[::-1], top.values[::-1])):
        ax.text(v+0.0003, i, f'{v:.4f}', va='center', color='white', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ─── TAB 3: Risk Calculator ───────────────────────────────────────────────────
elif nav == "🔍 Risk Calculator":
    st.title("🔍 Customer Churn Risk Calculator")
    st.markdown("Enter a customer's profile to generate their churn risk score.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📋 Demographics**")
        age        = st.slider("Age", 18, 92, 42)
        gender     = st.selectbox("Gender", ["Female", "Male"])
        geography  = st.selectbox("Geography", ["France", "Germany", "Spain"])
        credit_score = st.slider("Credit Score", 300, 900, 650)

    with col2:
        st.markdown("**💰 Financial Profile**")
        balance     = st.number_input("Account Balance (€)", 0, 300000, 75000, step=1000)
        salary      = st.number_input("Estimated Salary (€)", 10000, 250000, 100000, step=1000)
        num_prods   = st.selectbox("Number of Products", [1,2,3,4])
        has_cc      = st.selectbox("Has Credit Card", ["Yes","No"])

    with col3:
        st.markdown("**📊 Engagement**")
        is_active  = st.selectbox("Active Member", ["Yes","No"])
        tenure     = st.slider("Tenure (years)", 0, 10, 3)

    # Build feature vector
    row = {
        'CreditScore': credit_score, 'Age': age, 'Tenure': tenure,
        'Balance': balance, 'NumOfProducts': num_prods,
        'HasCrCard': 1 if has_cc=="Yes" else 0,
        'IsActiveMember': 1 if is_active=="Yes" else 0,
        'EstimatedSalary': salary,
        'Geography_France': 1 if geography=='France' else 0,
        'Geography_Germany': 1 if geography=='Germany' else 0,
        'Geography_Spain': 1 if geography=='Spain' else 0,
        'Gender_Female': 1 if gender=='Female' else 0,
        'Gender_Male': 1 if gender=='Male' else 0,
    }
    row['Balance_Salary_Ratio'] = int(float(balance) / (float(salary) + 1))
    row['Product_Density'] = int(float(num_prods) / (float(tenure) + 1))
    row['Engagement_Product'] = (1 if is_active=="Yes" else 0) * num_prods
    row['Age_Tenure_Interaction'] = age * tenure
    row['Zero_Balance'] = 1 if balance == 0 else 0
    row['Senior_Customer'] = 1 if age > 50 else 0

    X_input = pd.DataFrame([row])[feature_cols]
    rf_model = models['Random Forest']
    churn_prob = rf_model.predict_proba(X_input)[0][1]

    st.markdown("---")
    st.subheader("🎯 Risk Assessment")

    # Gauge
    c_gauge, c_info = st.columns([1.2, 1])
    with c_gauge:
        fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(polar=False))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
        # Simple bar gauge
        ax.barh([0], [1], color='#1a1a2e', height=0.4)
        color = '#ff4444' if churn_prob > 0.6 else ('#ff9900' if churn_prob > 0.3 else '#00cc44')
        ax.barh([0], [churn_prob], color=color, height=0.4)
        ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xlabel('Churn Probability', color='white')
        ax.text(churn_prob/2, 0, f'{churn_prob:.1%}', ha='center', va='center', color='white',
                fontsize=18, fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with c_info:
        if churn_prob > 0.6:
            st.markdown(f"""<div class="risk-high">
              <h3>🚨 HIGH RISK</h3>
              <p>Churn Probability: <strong>{churn_prob:.1%}</strong></p>
              <p>Immediate retention action recommended. Consider personalized offer or dedicated relationship manager contact.</p>
            </div>""", unsafe_allow_html=True)
        elif churn_prob > 0.3:
            st.markdown(f"""<div class="risk-medium">
              <h3>⚠️ MEDIUM RISK</h3>
              <p>Churn Probability: <strong>{churn_prob:.1%}</strong></p>
              <p>Monitor engagement closely. Targeted email campaigns and product upsell may help.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="risk-low">
              <h3>✅ LOW RISK</h3>
              <p>Churn Probability: <strong>{churn_prob:.1%}</strong></p>
              <p>Customer appears stable. Maintain engagement through loyalty rewards.</p>
            </div>""", unsafe_allow_html=True)

    # All model probabilities
    st.subheader("📊 Churn Probability Across All Models")
    probs = {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            Xi = scaler.transform(X_input)
        else:
            Xi = X_input
        probs[name] = model.predict_proba(Xi)[0][1]

    fig, ax = plt.subplots(figsize=(7,3))
    fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
    clrs = ['#4361ee','#f72585','#c77dff','#4cc9f0']
    bars = ax.barh(list(probs.keys()), list(probs.values()), color=clrs)
    ax.axvline(0.5, color='#aaa', linestyle='--', lw=1)
    ax.set_xlim(0,1)
    for bar,v in zip(bars, probs.values()):
        ax.text(v+0.01, bar.get_y()+bar.get_height()/2, f'{v:.1%}', va='center', color='white', fontweight='bold')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
    ax.set_xlabel('Churn Probability', color='white')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ─── TAB 4: What-If Simulator ─────────────────────────────────────────────────
elif nav == "🔄 What-If Simulator":
    st.title("🔄 What-If Churn Scenario Simulator")
    st.markdown("Compare two customer profiles side-by-side to observe churn risk changes.")

    col_base, col_alt = st.columns(2)

    def profile_inputs(col, label, default_active="No", default_prods=1, default_balance=0):
        with col:
            st.markdown(f"#### {label}")
            age     = st.slider(f"Age {label}", 18, 92, 45, key=f"age_{label}")
            balance = st.number_input(f"Balance {label} (€)", 0, 300000, default_balance, 1000, key=f"bal_{label}")
            prods   = st.selectbox(f"Products {label}", [1,2,3,4], index=default_prods-1, key=f"pr_{label}")
            active  = st.selectbox(f"Active {label}", ["Yes","No"], index=0 if default_active=="Yes" else 1, key=f"act_{label}")
            tenure  = st.slider(f"Tenure {label}", 0, 10, 3, key=f"ten_{label}")
            geo     = st.selectbox(f"Geography {label}", ["France","Germany","Spain"], key=f"geo_{label}")
            credit  = st.slider(f"Credit Score {label}", 300, 900, 650, key=f"cr_{label}")
            salary  = st.number_input(f"Salary {label}", 10000, 250000, 100000, 1000, key=f"sal_{label}")
            gender  = st.selectbox(f"Gender {label}", ["Female","Male"], key=f"gen_{label}")
        return age, balance, prods, active, tenure, geo, credit, salary, gender

    def build_row(age, balance, prods, active, tenure, geo, credit, salary, gender):
        r = {
            'CreditScore': credit, 'Age': age, 'Tenure': tenure, 'Balance': balance,
            'NumOfProducts': prods, 'HasCrCard': 1, 'IsActiveMember': 1 if active=="Yes" else 0,
            'EstimatedSalary': salary,
            'Geography_France': 1 if geo=='France' else 0,
            'Geography_Germany': 1 if geo=='Germany' else 0,
            'Geography_Spain': 1 if geo=='Spain' else 0,
            'Gender_Female': 1 if gender=='Female' else 0,
            'Gender_Male': 1 if gender=='Male' else 0,
        }
        r['Balance_Salary_Ratio'] = int(float(balance) / (float(salary) + 1))
        r['Product_Density'] = int(float(num_prods) / (float(tenure) + 1))
        r['Engagement_Product'] = (1 if active=="Yes" else 0) * num_prods
        r['Age_Tenure_Interaction'] = age * tenure
        r['Zero_Balance'] = 1 if balance==0 else 0
        r['Senior_Customer'] = 1 if age>50 else 0
        return r

    args_A = profile_inputs(col_base, "Profile A", "No", 1, 0)
    args_B = profile_inputs(col_alt, "Profile B", "Yes", 2, 80000)

    row_A = build_row(*args_A)
    row_B = build_row(*args_B)
    X_A = pd.DataFrame([row_A])[feature_cols]
    X_B = pd.DataFrame([row_B])[feature_cols]

    prob_A = models['Random Forest'].predict_proba(X_A)[0][1]
    prob_B = models['Random Forest'].predict_proba(X_B)[0][1]

    st.markdown("---")
    st.subheader("📊 Side-by-Side Risk Comparison")
    c1, c2 = st.columns(2)
    for col, label, prob in [(c1,"Profile A",prob_A),(c2,"Profile B",prob_B)]:
        color = "🔴" if prob>0.6 else ("🟡" if prob>0.3 else "🟢")
        with col:
            st.metric(f"{color} {label} Churn Probability", f"{prob:.1%}")

    delta = prob_B - prob_A
    if delta < 0:
        st.success(f"✅ Profile B reduces churn risk by **{abs(delta):.1%}** compared to Profile A")
    elif delta > 0:
        st.warning(f"⚠️ Profile B increases churn risk by **{delta:.1%}** compared to Profile A")
    else:
        st.info("Profiles have identical churn risk.")

    fig, ax = plt.subplots(figsize=(6,3))
    fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
    colors = ['#ff4444' if prob_A>0.6 else ('#ff9900' if prob_A>0.3 else '#00cc44'),
              '#ff4444' if prob_B>0.6 else ('#ff9900' if prob_B>0.3 else '#00cc44')]
    bars = ax.bar(['Profile A','Profile B'], [prob_A, prob_B], color=colors, width=0.4)
    ax.set_ylim(0,1); ax.axhline(0.5, color='#aaa', lw=1.5, linestyle='--', label='Risk Threshold')
    for bar,v in zip(bars,[prob_A,prob_B]):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f'{v:.1%}', ha='center', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ─── TAB 5: EDA ───────────────────────────────────────────────────────────────
elif nav == "📈 EDA Insights":
    st.title("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Balance Distribution by Churn")
        fig, ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
        for label, color in [(0,'#4cc9f0'),(1,'#f72585')]:
            subset = df_raw[df_raw['Exited']==label]['Balance']
            ax.hist(subset, bins=40, alpha=0.6, color=color,
                    label='Retained' if label==0 else 'Churned', edgecolor='none')
        ax.set_xlabel('Balance (€)', color='white'); ax.set_ylabel('Count', color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Credit Score by Churn")
        fig, ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
        for label, color in [(0,'#4cc9f0'),(1,'#f72585')]:
            subset = df_raw[df_raw['Exited']==label]['CreditScore']
            ax.hist(subset, bins=40, alpha=0.6, color=color,
                    label='Retained' if label==0 else 'Churned', edgecolor='none')
        ax.set_xlabel('Credit Score', color='white'); ax.set_ylabel('Count', color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🔗 Correlation Heatmap")
        num_cols = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
                    'HasCrCard','IsActiveMember','EstimatedSalary','Exited']
        fig, ax = plt.subplots(figsize=(7,6))
        fig.patch.set_facecolor('#0f0f1a')
        corr = df_raw[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdPu', ax=ax,
                    linewidths=0.3, annot_kws={'size':8}, cbar_kws={'shrink':0.8})
        ax.tick_params(colors='white', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col4:
        st.subheader("⏳ Churn Rate by Tenure")
        ten_ch = df_raw.groupby('Tenure')['Exited'].mean()
        ten_ch_values = np.asarray(ten_ch.values, dtype=float)
        fig, ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
        ax.plot(ten_ch.index, ten_ch_values, 'o-', color='#c77dff', lw=2, markersize=7)
        ax.fill_between(ten_ch.index, ten_ch_values, alpha=0.2, color='#c77dff')
        ax.set_xlabel('Tenure (years)', color='white'); ax.set_ylabel('Churn Rate', color='white')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.subheader("📊 Raw Dataset Sample")
    st.dataframe(df_raw.drop(columns=['AgeGroup'], errors='ignore').head(50), use_container_width=True)
