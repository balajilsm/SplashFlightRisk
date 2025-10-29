import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
CSV_PATH_DEFAULT = "/mnt/data/emp_history_data2.csv"

ACTIVE_FLAG_COL = "active_flag"     # 0 = still here, 1 = already left
ID_CANDIDATES = ["person_id", "employee_id", "name", "emp_no"]  # we try to show one of these in outputs

st.set_page_config(page_title="Flight Risk Predictor (Exit Risk Model)", layout="wide")
st.title("✈ Employee Exit Risk Predictor (Pure XGBoost, No sklearn)")

st.markdown("""
Goal: Learn from who ALREADY left, and predict who MIGHT leave next.

Process:
1. Load employee history
2. Train model using terminated vs active
3. Score current active employees (`active_flag = 0`)
4. Output:
   - `flight_risk_prediction` (probability to exit)
   - `flight_risk_band` (HIGH / MEDIUM / LOW / SAFE)
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

# Risk banding helper
def risk_band(score):
    if score >= 0.95:
        return "HIGH"
    elif score >= 0.90:
        return "MEDIUM"
    elif score >= 0.80:
        return "LOW"
    else:
        return "SAFE"

# ----------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------
st.header("1. Upload / Load Data")
uploaded = st.file_uploader("Upload employee CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded, dtype={'term_date': str, 'previous_salary': float}, low_memory=False)
    st.success("✅ File uploaded successfully.")
else:
    try:
        df = pd.read_csv(CSV_PATH_DEFAULT)
        st.info(f"No file uploaded — using default: {CSV_PATH_DEFAULT}")
    except Exception as e:
        st.error(f"Could not load CSV: {e}")
        st.stop()

st.subheader("Raw Data Preview")
st.dataframe(df.head(20))

# Validate column
if ACTIVE_FLAG_COL not in df.columns:
    st.error(f"Column '{ACTIVE_FLAG_COL}' not found in CSV. Please update ACTIVE_FLAG_COL.")
    st.stop()

# ----------------------------------------------------
# 2. FEATURE SELECTION
# ----------------------------------------------------
st.header("2. Choose Features for the Model")

# We DO NOT allow the label column as a feature
feature_cols_all = [c for c in df.columns if c not in [ACTIVE_FLAG_COL]]

# We'll propose defaults:
default_cols = []
# usually useful metadata columns for modeling
for col in [
    "tenure_years", "age", "age_group",
    "promotion_count", "move_count",
    "department_name", "position", "manager_name",
    "job_name", "job_group", "work_location_name",
    "location_state", "city",
    "previous_salary", "annual_salary",
    "percentage_change",
]:
    if col in feature_cols_all:
        default_cols.append(col)

# If default list ends up empty, just use everything except ACTIVE_FLAG_COL
if not default_cols:
    default_cols = feature_cols_all

selected_features = st.multiselect(
    "Select columns to TRAIN the model (attrition drivers):",
    options=feature_cols_all,
    default=default_cols
)

if len(selected_features) == 0:
    st.warning("Please select at least one feature.")
    st.stop()

run_model = st.button("🚀 Train & Score Exit Risk")
if not run_model:
    st.info("Click **Train & Score Exit Risk** to continue.")
    st.stop()

# ----------------------------------------------------
# 3. PREPARE TRAINING DATA
# ----------------------------------------------------
st.header("3. Train Exit Risk Model")

# label: who left (1) vs who is still here (0)
train_df = df.dropna(subset=[ACTIVE_FLAG_COL]).copy()

y_raw = train_df[ACTIVE_FLAG_COL]

# y_raw should already be 0 or 1:
# 0 = active, 1 = exited
# We keep that mapping exactly (1 means 'left'), which is good because
# we want model to predict "chance of becoming 1".
y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int)

# Features
X = train_df[selected_features]

# One-hot encode categoricals
X_encoded = pd.get_dummies(X, drop_first=True)

# Clean feature matrix
X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan).fillna(0)

# Remove all-zero columns (not useful for training)
if X_encoded.shape[1] > 0:
    X_encoded = X_encoded.loc[:, (X_encoded != 0).any(axis=0)]

# Split to train/test
X_train, X_test, y_train, y_test = simple_split(X_encoded, y, test_size=0.2, seed=42)

# Sanity checks
st.subheader("Training Diagnostics")
st.write("X_train shape:", X_train.shape)
st.write("Unique labels in y_train:", np.unique(y_train))
st.write("Label counts:", pd.Series(y_train).value_counts(dropna=False))
st.write("Any NaN in X_train?", np.isnan(X_train.values).any())

if X_train.empty:
    st.error("❌ Training data is empty after filtering.")
    st.stop()

if len(np.unique(y_train)) < 2:
    st.error("❌ Training label has only one class.\n"
             "You need both active_flag=0 (active) and active_flag=1 (terminated) rows in your data "
             "so the model can learn the difference.")
    st.stop()

# Convert to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train.values.astype(float), label=y_train.values.astype(float))
dtest  = xgb.DMatrix(X_test.values.astype(float),  label=y_test.values.astype(float))

params = {
    "objective": "binary:logistic",     # predicts probability of exit
    "eval_metric": "logloss",
    "eta": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

try:
    with st.spinner("Training XGBoost model on historical exit data..."):
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dtest, "test")],
            verbose_eval=False
        )
    st.success("✅ Model training complete!")
except xgb.core.XGBoostError as e:
    st.error("❌ XGBoost training failed:")
    st.code(str(e))
    st.stop()

# Evaluate model
y_test_pred_prob = model.predict(dtest)
y_test_pred_class = (y_test_pred_prob >= 0.5).astype(int)
test_accuracy = np.mean(y_test_pred_class == y_test.values)
st.write(f"**Validation Accuracy:** {test_accuracy:.3f}")
st.caption("This is how well the model separates 'stayed' vs 'left' on held-out historical data.")

# ----------------------------------------------------
# 4. SCORE CURRENT ACTIVE EMPLOYEES (WHO MAY LEAVE NEXT)
# ----------------------------------------------------
st.header("4. Predict Exit Risk for ACTIVE Employees")

current_active_df = df[df[ACTIVE_FLAG_COL] == 0].copy()

if current_active_df.empty:
    st.warning("No active employees found (active_flag == 0).")
    st.stop()

# Build feature matrix for current active
X_active = current_active_df[selected_features]

# One-hot encode using same transform as training
X_active = pd.get_dummies(X_active, drop_first=True)

# Align columns to training model columns
X_active = X_active.reindex(columns=X_encoded.columns, fill_value=0)

# Clean
X_active = X_active.replace([np.inf, -np.inf], np.nan).fillna(0)

dactive = xgb.DMatrix(X_active.values.astype(float))

# Predict probability of exit
current_active_df["flight_risk_prediction"] = model.predict(dactive)

# Banding
current_active_df["flight_risk_band"] = current_active_df["flight_risk_prediction"].apply(risk_band)

# Try to include an identifier in the final output
id_cols_available = [c for c in ID_CANDIDATES if c in current_active_df.columns]
display_cols = id_cols_available + selected_features + [
    "flight_risk_prediction",
    "flight_risk_band"
]
# remove dupes while keeping order
seen = set()
display_cols = [c for c in display_cols if not (c in seen or seen.add(c))]

st.subheader("Active Employee Exit Risk (sorted high → low)")
st.dataframe(
    current_active_df[display_cols]
    .sort_values("flight_risk_prediction", ascending=False)
)

# ----------------------------------------------------
# 5. RISK BAND SUMMARY / DRILLDOWN
# ----------------------------------------------------
st.header("5. Risk Band Summary")

band_counts = (
    current_active_df["flight_risk_band"]
    .value_counts()
    .reindex(["HIGH", "MEDIUM", "LOW", "SAFE"])
    .fillna(0)
    .astype(int)
)

st.bar_chart(band_counts)

st.write("Counts by band:", band_counts.to_dict())

chosen_band = st.selectbox(
    "Show employees in band:",
    ["HIGH", "MEDIUM", "LOW", "SAFE"]
)

band_view = current_active_df[current_active_df["flight_risk_band"] == chosen_band]

st.subheader(f"{chosen_band} Risk Employees")
if len(band_view) == 0:
    st.write("No employees in this band.")
else:
    st.dataframe(
        band_view[display_cols]
        .sort_values("flight_risk_prediction", ascending=False)
    )



st.success("Done ✅")
