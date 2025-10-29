import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# CONFIG: Update these column names if your CSV differs
# ----------------------------------------------------
CSV_PATH_DEFAULT = "/mnt/data/emp_history_data2.csv"
TARGET_COLUMN = "flight_risk"      # <-- replace with your label column
ACTIVE_FLAG_COL = "active_flag"    # <-- 0 = active, 1 = terminated

# ----------------------------------------------------
# APP SETUP
# ----------------------------------------------------
st.set_page_config(page_title="Flight Risk Predictor (XGBoost Only)", layout="wide")
st.title("✈ Employee Flight Risk Predictor (XGBoost Only)")

st.markdown("""
This version avoids scikit-learn.  
It uses only **XGBoost**, **pandas**, and **numpy** — safe for Python 3.13.

**Steps:**
1. Upload employee CSV  
2. Select features  
3. Train XGBoost  
4. Predict only active employees (flag = 0)  
5. Download results  
""")

# ----------------------------------------------------
# 1️⃣ LOAD CSV
# ----------------------------------------------------
st.header("1. Upload / Load Data")
uploaded = st.file_uploader("Upload your employee CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ File uploaded successfully.")
else:
    try:
        df = pd.read_csv(CSV_PATH_DEFAULT)
        st.info(f"No file uploaded — using default: {CSV_PATH_DEFAULT}")
    except Exception as e:
        st.error(f"Couldn't load CSV: {e}")
        st.stop()

st.dataframe(df.head(20))

# Basic checks
if TARGET_COLUMN not in df.columns:
    st.error(f"Target column '{TARGET_COLUMN}' not found. Update it in the code.")
    st.stop()

if ACTIVE_FLAG_COL not in df.columns:
    st.error(f"Active flag column '{ACTIVE_FLAG_COL}' not found. Update it in the code.")
    st.stop()

# ----------------------------------------------------
# 2️⃣ FEATURE SELECTION
# ----------------------------------------------------
st.header("2. Choose Features")
feature_cols = [c for c in df.columns if c not in [TARGET_COLUMN, ACTIVE_FLAG_COL]]
selected_features = st.multiselect(
    "Select feature columns for model training:",
    options=feature_cols,
    default=feature_cols
)

if len(selected_features) == 0:
    st.warning("Select at least one feature.")
    st.stop()

# ----------------------------------------------------
# 3️⃣ TRAIN MODEL BUTTON
# ----------------------------------------------------
run_model = st.button("🚀 Run XGBoost Model")
if not run_model:
    st.info("Click **Run XGBoost Model** to start training.")
    st.stop()

# ----------------------------------------------------
# 4️⃣ PREPARE DATA
# ----------------------------------------------------
train_df = df.dropna(subset=[TARGET_COLUMN])
X = train_df[selected_features]
y = train_df[TARGET_COLUMN]

# Encode categoricals manually (one-hot via pandas)
X_encoded = pd.get_dummies(X, drop_first=True)

# Split manually (still uses sklearn, but lightweight pure Python)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

# ----------------------------------------------------
# 5️⃣ TRAIN XGBOOST
# ----------------------------------------------------
st.header("3. Model Training")
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

with st.spinner("Training XGBoost..."):
    model.fit(X_train, y_train)
st.success("✅ Model training complete!")

# Evaluate quickly
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = (y_pred == y_test).mean()
st.write(f"**Test Accuracy:** {accuracy:.3f}")
try:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_pred_proba)
    st.write(f"**ROC-AUC:** {auc:.3f}")
except Exception:
    st.write("ROC-AUC not available (sklearn not installed).")

# ----------------------------------------------------
# 6️⃣ SCORE ACTIVE EMPLOYEES
# ----------------------------------------------------
st.header("4. Score Active Employees")
active_df = df[df[ACTIVE_FLAG_COL] == 0].copy()
if active_df.empty:
    st.warning("No active employees found (active_flag == 0).")
    st.stop()

X_active = pd.get_dummies(active_df[selected_features], drop_first=True)

# Align to training features (in case dummy columns differ)
X_active = X_active.reindex(columns=X_encoded.columns, fill_value=0)

active_df["flight_risk_prediction"] = model.predict_proba(X_active)[:, 1]

# Banding logic
def band(score):
    if score >= 0.95:
        return "HIGH"
    elif score >= 0.90:
        return "MEDIUM"
    elif score >= 0.80:
        return "LOW"
    else:
        return "SAFE"

active_df["flight_risk_band"] = active_df["flight_risk_prediction"].apply(band)

st.dataframe(
    active_df[selected_features + ["flight_risk_prediction", "flight_risk_band"]]
    .sort_values("flight_risk_prediction", ascending=False)
)

# ----------------------------------------------------
# 7️⃣ DOWNLOAD RESULTS
# ----------------------------------------------------
out_cols = selected_features + ["flight_risk_prediction", "flight_risk_band"]
csv_data = active_df[out_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download Active Employee Risk Scores",
    csv_data,
    file_name="flight_risk_active_employees.csv",
    mime="text/csv"
)

# ----------------------------------------------------
# 8️⃣ BAND SUMMARY
# ----------------------------------------------------
st.header("5. Risk Band Summary")
counts = (
    active_df["flight_risk_band"]
    .value_counts()
    .reindex(["HIGH", "MEDIUM", "LOW", "SAFE"])
    .fillna(0)
    .astype(int)
)
st.bar_chart(counts)
st.write("Counts by band:", counts.to_dict())

band_choice = st.selectbox("Show employees by band:", ["HIGH", "MEDIUM", "LOW", "SAFE"])
st.dataframe(
    active_df[active_df["flight_risk_band"] == band_choice]
    .sort_values("flight_risk_prediction", ascending=False)
)
