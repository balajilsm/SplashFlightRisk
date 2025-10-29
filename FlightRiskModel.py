import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
# CONFIG: EDIT THESE IF NEEDED
# -----------------------------
CSV_PATH_DEFAULT = "/mnt/data/emp_history_data2.csv"  # fallback if user doesn't upload
TARGET_COLUMN = "flight_risk"      # <-- change if your target label col is different
ACTIVE_FLAG_COL = "active_flag"    # <-- change if your active/terminated col is different (0=active,1=terminated)

st.set_page_config(
    page_title="Employee Flight Risk Predictor",
    layout="wide"
)

st.title("✈ Employee Flight Risk Predictor")

st.markdown("""
This app:
1. Reads employee history CSV  
2. Lets you choose which columns to train  
3. Trains XGBoost and Random Forest  
4. Scores only ACTIVE employees  
5. Outputs flight_risk_prediction and flight_risk_band  
""")

# -----------------------------
# 1. LOAD DATA
# -----------------------------
st.header("1. Upload / Load Data")

uploaded = st.file_uploader("Upload employee CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ File uploaded and loaded.")
else:
    try:
        df = pd.read_csv(CSV_PATH_DEFAULT)
        st.info(f"No file uploaded. Using default: {CSV_PATH_DEFAULT}")
    except Exception:
        st.error("No CSV available. Please upload a CSV.")
        st.stop()

st.subheader("Preview of Data")
st.dataframe(df.head(20))

# basic column validation before continuing
if TARGET_COLUMN not in df.columns:
    st.error(f"Target column '{TARGET_COLUMN}' not found in CSV. Please update TARGET_COLUMN in code.")
    st.stop()

if ACTIVE_FLAG_COL not in df.columns:
    st.error(f"Active flag column '{ACTIVE_FLAG_COL}' not found in CSV. Please update ACTIVE_FLAG_COL in code.")
    st.stop()

# -----------------------------
# 2. FEATURE SELECTION
# -----------------------------
st.header("2. Choose Features")

# don't allow using target or the active flag as features by default
all_features = [c for c in df.columns if c not in [TARGET_COLUMN, ACTIVE_FLAG_COL]]
default_features = all_features  # you can make this smaller if you want defaults

selected_features = st.multiselect(
    "Select feature columns for the model:",
    options=all_features,
    default=default_features
)

if len(selected_features) == 0:
    st.warning("Please select at least one feature to continue.")
    st.stop()

# model config UI (optional tuning knobs)
st.subheader("Training Options")
test_size = st.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5)
random_state = st.number_input("Random Seed", min_value=0, value=42, step=1)

# button to actually run training and prediction
run_model = st.button("🚀 Run Flight Risk Model")

if not run_model:
    st.info("👉 Click **Run Flight Risk Model** after selecting features.")
    st.stop()

# -----------------------------
# 3. PREPARE DATA
# -----------------------------
# Only keep rows where we have the target label to train
train_df = df.dropna(subset=[TARGET_COLUMN]).copy()

X = train_df[selected_features]
y = train_df[TARGET_COLUMN]

# detect numeric vs categorical
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in X.columns if col not in numeric_cols]

st.write("**Numeric columns detected:**", numeric_cols)
st.write("**Categorical columns detected:**", categorical_cols)

# preprocessing:
# numeric -> passthrough
# categorical -> one-hot
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ]
)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100.0,
    random_state=random_state,
    stratify=y if len(np.unique(y)) > 1 else None
)

# -----------------------------
# 4. DEFINE MODELS
# -----------------------------
rf_model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    ))
])

xgb_model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    ))
])

# -----------------------------
# 5. TRAIN + EVALUATE
# -----------------------------
st.header("3. Model Training Results")

def evaluate_model(name, model_pipeline):
    model_pipeline.fit(X_train, y_train)

    preds = model_pipeline.predict(X_test)

    # try to get probability for class 1 (leaving / high risk)
    if hasattr(model_pipeline.named_steps["clf"], "predict_proba"):
        prob = model_pipeline.predict_proba(X_test)[:, 1]
    else:
        prob = preds.astype(float)

    # classification report
    st.markdown(f"### {name}")
    st.text(classification_report(y_test, preds))

    # ROC AUC
    auc_score = None
    try:
        auc_score = roc_auc_score(y_test, prob)
    except Exception:
        pass

    st.write("ROC AUC:", auc_score)

    return {
        "name": name,
        "pipe": model_pipeline,
        "auc": auc_score
    }

rf_results = evaluate_model("Random Forest", rf_model)
xgb_results = evaluate_model("XGBoost", xgb_model)

# choose best model by AUC, fallback to RF if missing
if xgb_results["auc"] is not None and rf_results["auc"] is not None:
    best_model = xgb_results if xgb_results["auc"] >= rf_results["auc"] else rf_results
elif xgb_results["auc"] is not None:
    best_model = xgb_results
else:
    best_model = rf_results

st.success(f"🏆 Best model selected: {best_model['name']}")

# -----------------------------
# 6. SCORE ACTIVE EMPLOYEES
# -----------------------------
st.header("4. Score Active Employees")

# Filter only active people:
# active_flag == 0 means currently active (not terminated)
active_emp_df = df[df[ACTIVE_FLAG_COL] == 0].copy()

if active_emp_df.empty:
    st.warning("No active employees found (active_flag == 0). Check ACTIVE_FLAG_COL.")
else:
    # features for prediction
    active_X = active_emp_df[selected_features]

    # predict probability of leaving
    if hasattr(best_model["pipe"].named_steps["clf"], "predict_proba"):
        active_emp_df["flight_risk_prediction"] = best_model["pipe"].predict_proba(active_X)[:, 1]
    else:
        active_emp_df["flight_risk_prediction"] = best_model["pipe"].predict(active_X).astype(float)

    # banding logic
    def band_score(score):
        if score >= 0.95:
            return "HIGH"
        elif score >= 0.90:
            return "MEDIUM"
        elif score >= 0.80:
            return "LOW"
        else:
            return "SAFE"

    active_emp_df["flight_risk_band"] = active_emp_df["flight_risk_prediction"].apply(band_score)

    st.subheader("Active Employees with Flight Risk Scores (sorted high → low)")
    st.dataframe(
        active_emp_df[
            selected_features + ["flight_risk_prediction", "flight_risk_band"]
        ].sort_values("flight_risk_prediction", ascending=False)
    )

    # -----------------------------
    # 7. DOWNLOAD RESULTS
    # -----------------------------
    output_cols = selected_features + ["flight_risk_prediction", "flight_risk_band"]
    download_df = active_emp_df[output_cols].copy()

    csv_bytes = download_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Flight Risk Results (Active Employees)",
        data=csv_bytes,
        file_name="flight_risk_active_employees.csv",
        mime="text/csv"
    )

    # -----------------------------
    # 8. RISK BAND SUMMARY
    # -----------------------------
    st.header("5. Risk Band Summary")

    band_counts = (
        active_emp_df["flight_risk_band"]
        .value_counts()
        .reindex(["HIGH", "MEDIUM", "LOW", "SAFE"])
        .fillna(0)
        .astype(int)
    )

    st.bar_chart(band_counts)

    st.write("Counts by band:")
    st.write(band_counts)

    # drilldown
    chosen_band = st.selectbox(
        "Show employees in band:",
        ["HIGH", "MEDIUM", "LOW", "SAFE"]
    )

    band_view = active_emp_df[active_emp_df["flight_risk_band"] == chosen_band]
    st.subheader(f"{chosen_band} Risk Employees")
    st.dataframe(
        band_view[
            selected_features + ["flight_risk_prediction", "flight_risk_band"]
        ].sort_values("flight_risk_prediction", ascending=False)
    )
