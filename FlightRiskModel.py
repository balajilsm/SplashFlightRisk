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
# CONFIG (EDIT THESE NAMES IF NEEDED)
# -----------------------------
CSV_PATH_DEFAULT = "/mnt/data/emp_history_data2.csv"  # fallback if you don't upload
TARGET_COLUMN = "flight_risk"  # <-- change if your y column has a different name
ACTIVE_FLAG_COL = "active_flag"  # <-- change to your column. 0 = active, 1 = terminated

st.set_page_config(
    page_title="Employee Flight Risk Predictor",
    layout="wide"
)

st.title("✈ Employee Flight Risk Predictor")

st.markdown("""
This app:
1. Reads employee history CSV  
2. Lets you pick which feature columns to train  
3. Trains XGBoost and Random Forest  
4. Scores only *active* employees and assigns risk bands  
""")

# -----------------------------
# 1. LOAD DATA
# -----------------------------
st.header("1. Upload / Load Data")

uploaded = st.file_uploader("Upload employee CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("File uploaded and loaded.")
else:
    # fallback to server file path
    try:
        df = pd.read_csv(CSV_PATH_DEFAULT)
        st.info(f"No file uploaded. Using default: {CSV_PATH_DEFAULT}")
    except Exception as e:
        st.error("No CSV available. Please upload a CSV.")
        st.stop()

st.subheader("Preview of data")
st.dataframe(df.head(20))

# basic validation
if TARGET_COLUMN not in df.columns:
    st.error(f"Target column '{TARGET_COLUMN}' not found in CSV. Please update TARGET_COLUMN in code.")
    st.stop()

if ACTIVE_FLAG_COL not in df.columns:
    st.error(f"Active flag column '{ACTIVE_FLAG_COL}' not found in CSV. Please update ACTIVE_FLAG_COL in code.")
    st.stop()

# -----------------------------
# 2. FEATURE SELECTION UI
# -----------------------------
st.header("2. Choose Features")

all_features = [c for c in df.columns if c not in [TARGET_COLUMN]]
default_features = all_features

selected_features = st.multiselect(
    "Select feature columns for the model:",
    options=all_features,
    default=default_features
)
if len(selected_features) == 0:
    st.error("Please select at least one feature.")
    st.stop()

# drop rows with missing target
train_df = df.dropna(subset=[TARGET_COLUMN]).copy()

# X / y
X = train_df[selected_features]
y = train_df[TARGET_COLUMN]

# identify numeric vs categorical
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in X.columns if col not in numeric_cols]

st.write("**Numeric columns detected:**", numeric_cols)
st.write("**Categorical columns detected:**", categorical_cols)

# preprocessing:
# - pass numeric columns through (no scaling here for tree models)
# - one-hot encode categoricals
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# -----------------------------
# 3. TRAIN MODELS
# -----------------------------
st.header("3. Train Models")

test_size = st.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5)
random_state = st.number_input("Random Seed", min_value=0, value=42, step=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100.0,
    random_state=random_state,
    stratify=y if len(np.unique(y)) > 1 else None
)

# --- Model 1: Random Forest ---
rf_model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    ))
])

# --- Model 2: XGBoost ---
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

train_button = st.button("🚀 Train Models")

if train_button:
    st.subheader("Training...")

    # fit both models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # evaluate both
    def eval_model(name, pipe):
        preds = pipe.predict(X_test)
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
        else:
            # fallback if model has no predict_proba
            proba = preds.astype(float)

        auc = None
        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            pass

        st.markdown(f"### {name} Results")
        st.text(classification_report(y_test, preds))
        st.write("ROC AUC:", auc)

        return {
            "name": name,
            "pipe": pipe,
            "auc": auc,
        }

    rf_results = eval_model("Random Forest", rf_model)
    xgb_results = eval_model("XGBoost", xgb_model)

    # pick best model by AUC (fallback to RF if AUC missing)
    if xgb_results["auc"] is not None and rf_results["auc"] is not None:
        best_model = xgb_results if xgb_results["auc"] >= rf_results["auc"] else rf_results
    elif xgb_results["auc"] is not None:
        best_model = xgb_results
    else:
        best_model = rf_results

    st.success(f"Best model selected: {best_model['name']}")

    # -----------------------------
    # 4. SCORE ACTIVE EMPLOYEES ONLY
    # -----------------------------
    st.header("4. Score Active Employees")

    # active employees = flag 0 (not terminated)
    active_emp_df = df[df[ACTIVE_FLAG_COL] == 0].copy()

    # keep only selected features for prediction
    active_features = active_emp_df[selected_features]

    # get probability of leaving (class 1)
    if hasattr(best_model["pipe"].named_steps["clf"], "predict_proba"):
        active_emp_df["flight_risk_prediction"] = best_model["pipe"].predict_proba(active_features)[:, 1]
    else:
        active_emp_df["flight_risk_prediction"] = best_model["pipe"].predict(active_features).astype(float)

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

    # show result
    st.subheader("Active Employees with Flight Risk Scores")
    st.dataframe(
        active_emp_df[
            selected_features
            + ["flight_risk_prediction", "flight_risk_band"]
        ].sort_values("flight_risk_prediction", ascending=False)
    )

    # download CSV
    output_cols = selected_features + ["flight_risk_prediction", "flight_risk_band"]
    download_df = active_emp_df[output_cols]

    csv_bytes = download_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Flight Risk Results (Active Employees)",
        data=csv_bytes,
        file_name="flight_risk_active_employees.csv",
        mime="text/csv"
    )

    # -----------------------------
    # 5. RISK BANDS SUMMARY
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

    # drill-down by clicking each band (simple selectbox)
    chosen_band = st.selectbox(
        "Show employees in band:",
        ["HIGH", "MEDIUM", "LOW", "SAFE"]
    )

    band_view = active_emp_df[active_emp_df["flight_risk_band"] == chosen_band]
    st.dataframe(
        band_view[
            selected_features
            + ["flight_risk_prediction", "flight_risk_band"]
        ].sort_values("flight_risk_prediction", ascending=False)
    )

else:
    st.info("Select features, then click '🚀 Train Models'.")
