import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
CSV_PATH_DEFAULT = "/mnt/data/emp_history_data2.csv"
TARGET_COLUMN = "flight_risk"      # your label column (0/1)
ACTIVE_FLAG_COL = "active_flag"    # 0 = active, 1 = terminated

# ----------------------------------------------------
# STREAMLIT HEADER
# ----------------------------------------------------
st.set_page_config(page_title="Flight Risk Predictor (Pure XGBoost)", layout="wide")
st.title("✈ Employee Flight Risk Predictor (Pure XGBoost)")

st.markdown("""
This version is 100% compatible with **Python 3.13** — no scikit-learn needed.  
It uses the core XGBoost API directly.

**Workflow**
1. Upload employee CSV  
2. Choose features  
3. Train XGBoost model  
4. Predict for active employees (flag = 0)  
5. Download results with risk bands  
""")

# ----------------------------------------------------
# Helper: simple train/test split
# ----------------------------------------------------
def simple_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )

# ----------------------------------------------------
# 1️⃣ LOAD DATA
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
        st.error(f"Could not load CSV: {e}")
        st.stop()

st.dataframe(df.head(20))

# Validate
if TARGET_COLUMN not in df.columns:
    st.error(f"Target column '{TARGET_COLUMN}' not found.")
    st.stop()
if ACTIVE_FLAG_COL not in df.columns:
    st.error(f"Active flag column '{ACTIVE_FLAG_COL}' not found.")
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
    st.warning("Please select at least one feature.")
    st.stop()

# ----------------------------------------------------
# 3️⃣ TRAIN BUTTON
# ----------------------------------------------------
run_model = st.button("🚀 Run XGBoost Model")
if not run_model:
    st.info("Click **Run XGBoost Model** to start training.")
    st.stop()

# ----------------------------------------------------
# 4️⃣ PREPARE & CLEAN DATA
# ----------------------------------------------------
train_df = df.dropna(subset=[TARGET_COLUMN])
X = train_df[selected_features]
y = train_df[TARGET_COLUMN]

# Convert y to numeric (0/1) safely
y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

# One-hot encode categoricals & clean
X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan).fillna(0)

# Split data
X_train, X_test, y_train, y_test = simple_split(X_encoded, y, test_size=0.2, seed=42)

# Diagnostic info
st.write("Training shape:", X_train.shape)
st.write("Unique labels:", np.unique(y_train))
st.write("Any NaN in X_train?", np.isnan(X_train.values).any())

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
dtest = xgb.DMatrix(X_test.values, label=y_test.values)

# ----------------------------------------------------
# 5️⃣ TRAIN PURE XGBOOST MODEL
# ----------------------------------------------------
st.header("3. Model Training")

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

with st.spinner("Training XGBoost model..."):
    model = xgb.train(params, dtrain, num_boost_round=300, evals=[(dtest, "test")], verbose_eval=False)

st.success("✅ Model training complete!")

# Evaluate
y_pred = model.predict(dtest)
y_pred_class = (y_pred >= 0.5).astype(int)
accuracy = np.mean(y_pred_class == y_test.values)
st.write(f"**Test Accuracy:** {accuracy:.3f}")

# ----------------------------------------------------
# 6️⃣ SCORE ACTIVE EMPLOYEES
# ----------------------------------------------------
st.header("4. Score Active Employees")
active_df = df[df[ACTIVE_FLAG_COL] == 0].copy()

if active_df.empty:
    st.warning("No active employees found (active_flag == 0).")
    st.stop()

# Prepare active data
X_active = pd.get_dummies(active_df[selected_features], drop_first=True)
X_active = X_active.reindex(columns=X_encoded.columns, fill_value=0)
dactive = xgb.DMatrix(X_active.values)

# Predict
active_df["flight_risk_prediction"] = model.predict(dactive)

# Banding logic
def risk_band(score):
    if score >= 0.95:
        return "HIGH"
    elif score >= 0.90:
        return "MEDIUM"
    elif score >= 0.80:
        return "LOW"
    else:
        return "SAFE"

active_df["flight_risk_band"] = active_df["flight_risk_prediction"].apply(risk_band)

# Show results
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
    label="⬇ Download Active Employee Risk Scores",
    data=csv_data,
    file_name="flight_risk_active_employees.csv",
    mime="text/csv"
)

# ----------------------------------------------------
# 8️⃣ RISK BAND SUMMARY
# ----------------------------------------------------
st.header("5. Risk Band Summary")
band_counts = (
    active_df["flight_risk_band"]
    .value_counts()
    .reindex(["HIGH", "MEDIUM", "LOW", "SAFE"])
    .fillna(0)
    .astype(int)
)
st.bar_chart(band_counts)
st.write("Counts by band:", band_counts.to_dict())

band_choice = st.selectbox("Show employees in band:", ["HIGH", "MEDIUM", "LOW", "SAFE"])
st.dataframe(
    active_df[active_df["flight_risk_band"] == band_choice]
    .sort_values("flight_risk_prediction", ascending=False)
)
