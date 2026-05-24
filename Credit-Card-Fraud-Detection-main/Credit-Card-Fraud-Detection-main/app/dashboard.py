"""
Streamlit interactive visual analytics dashboard.
Enables real-time fraud scoring, interactive feature adjustment, and model comparisons.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

from src.predict import FraudPredictor
from src.exceptions import InferenceError

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Center",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment configs
model_path = os.getenv("MODEL_PATH", "models/random_forest_champion/random_forest_champion.joblib")


# Cached helper to load the prediction engine
@st.cache_resource
def get_predictor(path: str) -> FraudPredictor:
    return FraudPredictor(path)


# Title header
st.title("💳 Credit Card Fraud Detection Center")
st.markdown(
    "Interactive MLOps dashboard showcasing real-time transaction screening, "
    "historical exploratory insights, and baseline-vs-champion model metrics."
)
st.write("---")

# Sidebar navigation
st.sidebar.title("Navigation Panel")
app_mode = st.sidebar.radio(
    "Choose Mode:",
    ["Overview & Analytics", "Interactive Risk Predictor", "Model Metrics & Reports"]
)

# Sidebar model details card
st.sidebar.write("---")
st.sidebar.subheader("Model Status")

if os.path.exists(model_path):
    try:
        predictor = get_predictor(model_path)
        st.sidebar.success("🟢 Active model loaded!")
        
        # Load companion metadata details
        meta_path = model_path.replace(".joblib", "_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            st.sidebar.info(
                f"**Name:** {meta.get('model_name')}\n\n"
                f"**Type:** {meta.get('type')}\n\n"
                f"**Saved At:** {meta.get('saved_at', 'N/A')[:10]}"
            )
    except Exception as e:
        st.sidebar.error(f"🔴 Model load failure: {e}")
else:
    st.sidebar.warning("⚠️ Champion model binary not found. Please run 'train' command in CLI.")

# ----------------- Mode 1: Overview & Analytics -----------------
if app_mode == "Overview & Analytics":
    st.header("📊 Transaction Dataset Insights & Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Profiles & Explanations")
        st.markdown(
            """
            This system evaluates cardholder patterns using **highly anonymous transaction datasets**.
            *   **PCA Dimensional Components ($V_1$ to $V_{28}$):** Features extracted via Principal Component Analysis 
                to safeguard private client indices while retaining transaction variances.
            *   **Amount (€):** The transaction volume in Euros. Heavy-tailed with intense outlier metrics.
            *   **Time (s):** Seconds elapsed since the first recorded epoch.
            
            **The Imbalance Penalty:**
            Fraud occupies a mere **0.172%** of raw transactions. Traditional standard classifiers defaulted 
            to majority prediction, failing to detect fraud while claiming 99.8% superficial accuracy. 
            This pipeline leverages **SMOTE (Synthetic Minority Over-sampling Technique)** inside leak-proof CV 
            folds to establish robust decision boundaries.
            """
        )
        
        st.subheader("Upload Custom Transaction CSV for Bulk Prediction")
        uploaded_file = st.file_uploader("Upload CSV transaction records", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"Ingested {len(df_upload)} records successfully!")
                
                if os.path.exists(model_path):
                    with st.spinner("Processing batch scoring..."):
                        preds, probs = predictor.predict_batch(df_upload)
                    
                    df_upload["Prediction"] = ["Fraud" if p == 1 else "Legit" for p in preds]
                    df_upload["Fraud Probability"] = probs
                    
                    fraud_count = int(np.sum(preds))
                    st.metric(label="Flagged Fraudulent Transactions", value=f"{fraud_count} / {len(preds)}")
                    
                    st.dataframe(
                        df_upload[["Time", "Amount", "Prediction", "Fraud Probability"]].head(15),
                        use_container_width=True
                    )
                else:
                    st.warning("Model binary is not loaded. Cannot run bulk scores.")
            except Exception as e:
                st.error(f"Failed to process CSV: {e}")

    with col2:
        st.subheader("Exploratory Data Visualizations")
        # Render static assets if they exist
        dist_img_path = "visuals/fraud_distribution.png"
        corr_img_path = "visuals/correlation_matrix.png"
        
        if os.path.exists(dist_img_path):
            st.image(dist_img_path, caption="Figure 1: Fraud Imbalance & Amount KDE Distributions", use_column_width=True)
        else:
            st.info("EDA Distribution plot is not rendered yet. Execute 'train' CLI step to generate.")
            
        if os.path.exists(corr_img_path):
            st.image(corr_img_path, caption="Figure 2: Pearson Feature Correlations Map", use_column_width=True)
        else:
            st.info("Correlation heatmap asset is missing. Execute 'train' CLI step to generate.")

# ----------------- Mode 2: Interactive Risk Predictor -----------------
elif app_mode == "Interactive Risk Predictor":
    st.header("⚡ Real-Time Transaction Screening & Risk Estimator")
    st.markdown("Adjust transaction feature sliders in real-time to compute fraud probabilities instantly.")
    
    if not os.path.exists(model_path):
        st.error("No active model loaded. Risk scoring is currently offline. Please run the CLI training command.")
    else:
        # Create structural layout for input controls
        col_inputs1, col_inputs2, col_inputs3 = st.columns(3)
        
        with col_inputs1:
            st.subheader("Metadata & Scale")
            amount = st.slider("Transaction Amount (€)", min_value=0.0, max_value=1000.0, value=85.00, step=1.0)
            time = st.slider("Transaction Time (Sec)", min_value=0, max_value=172800, value=50000, step=100)
            
            st.subheader("Primary PCA Features")
            v1 = st.slider("V1 (Time Profile Variance)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            v2 = st.slider("V2 (Geo Location Cluster)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            v3 = st.slider("V3 (Merchant Code Cluster)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            v4 = st.slider("V4 (Terminal Risk Index)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            
        with col_inputs2:
            st.subheader("High-Correlation PCA Features")
            v10 = st.slider("V10 (User Session Behavior)", min_value=-10.0, max_value=5.0, value=0.0, step=0.1)
            v12 = st.slider("V12 (Velocity Threshold Rate)", min_value=-10.0, max_value=5.0, value=0.0, step=0.1)
            v14 = st.slider("V14 (Card Presence Variance)", min_value=-10.0, max_value=5.0, value=0.0, step=0.1)
            v17 = st.slider("V17 (International Routing Shift)", min_value=-10.0, max_value=5.0, value=0.0, step=0.1)
            
            # Supplementary slider indices
            v5 = st.slider("V5", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            v6 = st.slider("V6", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            v7 = st.slider("V7", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

        with col_inputs3:
            st.subheader("Supporting PCA Features")
            v8 = st.slider("V8", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
            v9 = st.slider("V9", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
            v11 = st.slider("V11", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
            v13 = st.slider("V13", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
            v15 = st.slider("V15", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
            v16 = st.slider("V16", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
            
            # Default other variables (V18-V28) to 0.0
            other_vs = {f"V{i}": 0.0 for i in range(18, 29)}
            
        # Group inputs together
        input_record = {
            "Time": float(time),
            "Amount": float(amount),
            "V1": float(v1),
            "V2": float(v2),
            "V3": float(v3),
            "V4": float(v4),
            "V5": float(v5),
            "V6": float(v6),
            "V7": float(v7),
            "V8": float(v8),
            "V9": float(v9),
            "V10": float(v10),
            "V11": float(v11),
            "V12": float(v12),
            "V13": float(v13),
            "V14": float(v14),
            "V15": float(v15),
            "V16": float(v16),
            "V17": float(v17),
            **other_vs
        }
        
        st.write("---")
        
        # Scoring Execution section
        score_col1, score_col2 = st.columns([1, 2])
        
        with score_col1:
            st.subheader("Transaction Summary")
            st.info(f"**Amount:** €{amount:.2f}\n\n**Time Count:** {time} sec")
            run_score = st.button("🔴 SCREEN TRANSACTION FOR RISK", use_container_width=True)
            
        with score_col2:
            st.subheader("Inference Result")
            if run_score:
                try:
                    result = predictor.predict_single(input_record)
                    prob = result["fraud_probability"]
                    
                    st.write(f"**Calculated Fraud Probability Score:** `{prob*100:.2f}%`")
                    st.progress(prob)
                    
                    if result["prediction_class"] == 1:
                        st.error(
                            f"🚨 **CRITICAL WARNING:** This transaction is flagged as **{result['prediction_label'].upper()}**!\n\n"
                            f"Risk threshold surpassed. Instantly freeze card authorization."
                        )
                    else:
                        st.success(
                            f"✅ **TRANSACTION APPROVED:** This transaction is marked as **{result['prediction_label'].upper()}**.\n\n"
                            f"Risk parameters fall within acceptable boundaries."
                        )
                except InferenceError as e:
                    st.error(f"Inference Engine Crash: {e.message}")
            else:
                st.info("Click the screening button to score parameters through the Random Forest model.")

# ----------------- Mode 3: Model Metrics & Reports -----------------
elif app_mode == "Model Metrics & Reports":
    st.header("📈 Model Benchmarks & Comparison Summary")
    
    # Render report details if they exist in reports/
    lr_report_path = "reports/logistic_regression_baseline_report.json"
    rf_report_path = "reports/random_forest_champion_report.json"
    
    if os.path.exists(lr_report_path) and os.path.exists(rf_report_path):
        with open(lr_report_path, "r") as f:
            lr_data = json.load(f)
        with open(rf_report_path, "r") as f:
            rf_data = json.load(f)
            
        # Metric comparison table
        st.subheader("Historical Performance Benchmark Matrix")
        
        metrics_comp = []
        for k in lr_data["metrics"].keys():
            metrics_comp.append({
                "Metric": k,
                "Logistic Regression (Baseline)": f"{lr_data['metrics'][k]*100:.2f}%" if "AUC" not in k and "MCC" not in k else f"{lr_data['metrics'][k]:.4f}",
                "Random Forest (Champion)": f"{rf_data['metrics'][k]*100:.2f}%" if "AUC" not in k and "MCC" not in k else f"{rf_data['metrics'][k]:.4f}",
            })
            
        st.table(pd.DataFrame(metrics_comp))
    else:
        st.warning("Performance metrics report assets not found. Execute 'evaluate' CLI step to compute.")
        
    st.write("---")
    
    # Renders evaluation plots
    st.subheader("Model Validation Heatmaps and Threshold Curves")
    
    rf_cm_path = "visuals/random_forest_champion_confusion_matrix.png"
    rf_curves_path = "visuals/random_forest_champion_evaluation_curves.png"
    
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        if os.path.exists(rf_cm_path):
            st.image(rf_cm_path, caption="Random Forest Champion: Confusion Matrix heatmap", use_column_width=True)
        else:
            st.info("Confusion Matrix image missing. Run 'evaluate' command to generate.")
            
    with plot_col2:
        if os.path.exists(rf_curves_path):
            st.image(rf_curves_path, caption="Random Forest Champion: ROC & Precision-Recall curves", use_column_width=True)
        else:
            st.info("Evaluation curves plot is missing. Run 'evaluate' command to generate.")
