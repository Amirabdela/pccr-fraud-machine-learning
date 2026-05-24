# 💳 Production-Grade Credit Card Fraud Detection Pipeline

An end-to-end, modular, and mathematically rigorous machine learning pipeline designed to detect fraudulent credit card transactions on highly imbalanced datasets. 

This repository transitions a typical exploratory Jupyter notebook into a **production-ready MLOps codebase** adhering to standard software engineering patterns (clean architecture, strict type hinting, robust exception boundaries, structured logging, and configurability) and robust academic guidelines (leakage-free preprocessing, baseline-vs-champion models, and imbalanced classification metrics).

---

## 1. Executive Summary & Problem Statement

In transactional financial networks, fraudulent transactions represent a minute fraction of the total volume—typically **$< 0.2\%$**. Standard machine learning models trained on raw imbalanced datasets fail to catch fraud: because legitimates represent $99.8\%$ of the split, a trivial predictor that labels *every* transaction as legitimate achieves an appealing **$99.8\%$ accuracy** while completely failing to detect actual fraud.

This pipeline provides a scientifically rigorous approach:
*   **The Baseline Model:** A standardized, scaled **Logistic Regression** classifier equipped with balanced class weights to serve as a linear baseline.
*   **The Champion Model:** An ensemble **Random Forest Classifier** trained using **SMOTE** (Synthetic Minority Over-sampling Technique) to expand decision boundaries near positive clusters.
*   **The Core Engineering Priority:** Preventing **Data Leakage**. Oversampling must *never* bleed into test partitions. SMOTE is encapsulated strictly inside an `imblearn` pipeline so that oversampling occurs *only* within cross-validation folds on training splits, maintaining the test partition as an untouched, realistic benchmark.

---

## 2. Repository Folder Architecture

The project is structured as a professional Python library and MLOps repository:

```text
Credit-Card-Fraud-Detection/
│
├── config/
│   ├── config.yaml               # Centralized hyperparameters, paths, and preprocessing options
│   └── logging_config.yaml       # Configuration for rolling log files and console formats
│
├── data/
│   ├── raw/                      # raw dataset directory (contains creditcard.csv or gitkeep)
│   └── processed/                # preprocessed intermediate dataset partitions
│
├── app/                          # Production Web Interfaces
│   ├── main_api.py               # FastAPI inference microservice (FastAPI + Pydantic)
│   └── dashboard.py              # Streamlit interactive analytics dashboard
│
├── src/                          # Modular Python package
│   ├── __init__.py
│   ├── config.py                 # Configuration manager merging yaml and .env settings
│   ├── exceptions.py             # Custom domain exceptions (DataIngestionError, PreprocessingError, etc.)
│   ├── data_loader.py            # Dataset loader with high-fidelity synthetic fallback
│   ├── preprocessing.py          # Imputation, stratified partitioning, and ColumnTransformers
│   ├── train.py                  # Stratified K-Fold CV training and tuning
│   ├── evaluate.py               # Precision-Recall, MCC calculation, Markdown/JSON report export
│   ├── predict.py                # Type-hinted batch and real-time predictor class
│   ├── visualize.py              # Publication-grade headless visualization module
│   └── utils.py                  # Logging configuration and joblib companion serialization
│
├── tests/                        # Automated unit and integration test suite (pytest)
│   ├── __init__.py
│   ├── test_data_loader.py       # Validates loading, schemas, and fallback generation
│   ├── test_preprocessing.py     # Asserts stratified shapes, Winsorization, and scaling pass-throughs
│   └── test_pipeline.py          # Asserts leakage-free SMOTE and serialization cycles
│
├── models/                       # Versioned model joblib binaries and metadata JSONs
├── reports/                      # Auto-generated Markdown & JSON performance summaries
├── visuals/                      # Exported evaluation charts and heatmaps
├── logs/                         # Rolling pipeline execution log files
│
├── Dockerfile                    # Containerization instructions
├── docker-compose.yml            # Multi-service composition (FastAPI API + Streamlit Dashboard)
├── .github/workflows/ci.yml      # CI/CD pipeline running pytest automatically on pushes
├── .env.example                  # Environment configuration template
├── requirements.txt              # Production pinned requirements (Lightweight Core)
└── main.py                       # Unified Click-based CLI orchestrator entrypoint
```

---

## 3. Setup & Installation

### Option A: Standard Lightweight Core (Recommended)
This installs only the essential ML dependencies, bypassing heavy web frameworks. It allows you to run training, testing, and evaluation pipelines instantly out-of-the-box:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Muhammad-Talha4k/Credit-Card-Fraud-Detection.git
    cd Credit-Card-Fraud-Detection
    ```

2.  **Establish a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Lightweight Core Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset (Optional):**
    Place `creditcard.csv` from Kaggle into `data/raw/creditcard.csv`.
    > 💡 **Out-of-the-Box Fallback:** If you do not download the dataset, the loader will **automatically generate a high-fidelity synthetic dataset (15,000 samples)** matching Kaggle's exact feature scales, PCA properties, and fraud ratio. This guarantees the entire pipeline runs flawlessly immediately without any manual configuration!

---

## 4. Run the Project from End-to-End

The pipeline is orchestrated via a unified Click command CLI in `main.py`.

### Step 1: Run the Automated Verifications
Verify model logic, stratified distributions, and leakage prevention assertions using the pytest suite:
```bash
python -m pytest tests/
```

### Step 2: Train the Pipeline
Ingest data, cap extreme outliers (Winsorization), generate Pearson correlation matrices, partition stratified splits, perform Stratified 5-Fold Cross-Validation, train both Logistic Regression and SMOTE Random Forest models, and serialize binaries:
```bash
python main.py train --quick-mode
```

### Step 3: Evaluate Performance & Generate Reports
Compute balanced evaluation metrics (Precision, Recall, F1, MCC, ROC-AUC, PR-AUC), generate Confusion Matrices, plot evaluation curves, and automatically export high-quality JSON and Markdown performance reports to the `reports/` folder:
```bash
python main.py evaluate
```

### Step 4: Run Live Mock Predictions
Perform batch scoring and real-time transaction risk screening:
```bash
python main.py predict
```

### Step 5: Run Web Interfaces (Optional Extra)
To launch the FastAPI microservice and Streamlit visual dashboard, first install the extra requirements:
```bash
pip install fastapi uvicorn streamlit
```
Spin up the FastAPI real-time scoring microservice in a terminal:
```bash
python main.py serve
```
Spin up the Streamlit interactive metrics and manual sliding risk estimator dashboard in a separate terminal:
```bash
streamlit run app/dashboard.py
```

---

## 5. Model Performance Comparison

Below are the actual empirical results obtained on the testing split (3,000 samples):

| Evaluation Metric | Baseline Model (Logistic Regression) | Champion Model (Random Forest + SMOTE) | Academic & Financial Significance |
| :--- | :---: | :---: | :--- |
| **Accuracy** | $99.97\%$ | $99.87\%$ | Total percentage of correct labels. Highly misleading. |
| **Precision** | $100.00\%$ | $100.00\%$ | Out of all flagged alerts, how many are true frauds. Controls operational audit friction. |
| **Recall (Sensitivity)** | $80.00\%$ | $20.00\%$ | Out of all actual frauds, how many are caught. Controls financial loss prevention. |
| **F1-Score** | $0.8889$ | $0.3333$ | Harmonic mean of Precision and Recall. Balances both dimensions. |
| **Matthews Correlation (MCC)** | **$0.8943$** | $0.4469$ | **Balanced indicator.** Evaluates all four confusion matrix quadrants. |
| **ROC-AUC** | $1.0000$ | $0.9996$ | Overall trade-off between TPR and FPR. |
| **PR-AUC** | **$1.0000$** | **$0.8855$** | **Primary metric.** Area under Precision-Recall Curve. Crucial for heavy minority splits. |

**Key Academic Observations:**
On our high-fidelity synthetic demo partitions, the baseline scaled **Logistic Regression** performs with excellent precision and recall ($80\%$). Due to the regularized forest depth in quick mode, the baseline holds champion standing. This demonstrates that standard linear estimators with robust outlier capping (Winsorization) and feature standardizations often serve as formidable baseline champions, showing why thorough pipeline comparative studies are highly valued in research.

---

## 6. MLOps Upgrades & Best Practices

To transition this codebase into an industry-grade system, several best practices were integrated:
1.  **Strict Config Isolation:** Hyperparameters, capping limits, split sizes, and paths are isolated cleanly in `config/config.yaml`.
2.  **Domain Custom Exceptions:** Explicit custom errors (`DataIngestionError`, `PreprocessingError`, `ModelTrainingError`, `InferenceError`) with tailored try/except blocks replace standard python crashes.
3.  **Comprehensive Type Hinting:** Adheres strictly to PEP 484 type annotations for parameters and return types across all functions.
4.  **Multi-Handler Logging:** Console stdout logging is kept clean, while rolling files (`logs/pipeline.log`, `logs/error.log`) capture detailed file and line markers.
5.  **Metadata Serialization:** Serialized model binaries (`.joblib`) are accompanied by a companion `metadata.json` capturing the exact saved timestamp, metrics, and parameters, ensuring strict reproducibility.

---

## 7. Future Directions

*   **Cost-Sensitive Learning:** Introduce a custom objective function that mathematically penalizes False Negatives (missed fraud) more severely than False Positives.
*   **Time-Series Feature Engineering:** Incorporate rolling temporal features (e.g., transaction frequency within the last 1 hour) to capture behavior drift.
*   **Microservice Deployment:** Spin up the FastAPI API (`app/main_api.py`) or the Streamlit Dashboard (`app/dashboard.py`) using the pre-configured `Dockerfile` and `docker-compose.yml`.

---

## 8. References

1.  Dal Pozzolo, A., et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*.
2.  Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*.
3.  [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
