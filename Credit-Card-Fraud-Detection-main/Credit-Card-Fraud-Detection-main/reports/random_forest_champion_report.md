# Performance Evaluation Report: random_forest_champion

Auto-generated on the testing partition.

## Executive Metrics Summary

| Evaluation Metric | Score | Scientific Relevance for Fraud Detection |
| :--- | :---: | :--- |
| **Accuracy** | 0.9987 | Percentage of correct classifications. **Highly misleading** due to major class imbalance. |
| **Precision** | 1.0000 | Ratio of true frauds to all flagged anomalies. Controls operational audit friction. |
| **Recall (Sensitivity)** | 0.2000 | Percentage of actual frauds intercepted. Controls financial loss prevention. |
| **F1-Score** | 0.3333 | Harmonic mean of Precision and Recall. Balances both dimensions. |
| **Matthews Correlation Coefficient (MCC)** | 0.4469 | Balanced correlation measure ranging from -1 to +1. High score indicates a perfect predictor. |
| **ROC-AUC** | 0.9996 | Area Under ROC Curve. Sensitivity vs False Positive Rate trade-offs. |
| **PR-AUC** | 0.8855 | **Primary Metric.** Area Under Precision-Recall Curve. Crucial for heavy minority class splits. |

---

## Confusion Matrix Analysis

```
                  Predicted Legitimate    Predicted Fraudulent
Actual Legitimate       2995       (TN)           0          (FP)
Actual Fraudulent       4          (FN)           1          (TP)
```

*   **Total Transactions Analyzed:** 3000
*   **Actual Fraud Cases:** 5
*   **Successfully Intercepted (True Positives):** 1 (20.00%)
*   **Missed Fraud (False Negatives - Financial Loss):** 4
*   **False Alarms (False Positives - Audit Friction):** 0

---

## Academic Formulas and Reference

1.  **Matthews Correlation Coefficient (MCC):**
    $$\text{MCC} = \frac{\text{TP} \times \text{TN} - \text{FP} \times \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$
    
2.  **Precision-Recall Area Under Curve (PR-AUC):**
    Integrates the Precision ($P$) as a function of Recall ($R$):
    $$\text{PR-AUC} = \int_0^1 P(R) \, dR$$
