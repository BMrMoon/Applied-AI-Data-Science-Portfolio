# Evaluation of Classification Models: Churn & Fraud Detection 🛡️🔍

This project focuses on the critical evaluation of classification models using performance metrics beyond simple accuracy. It covers two distinct scenarios: **Customer Churn Prediction** and **Banking Fraud Detection**, highlighting the importance of Confusion Matrix analysis in imbalanced datasets.

---

## 🎯 Project Objectives
- **Metric Computation:** Manually calculating and interpreting Accuracy, Precision, Recall, and F1-Score.
- **Problem Diagnosis:** Analyzing why a high-accuracy model (90.5%) can fail in a real-world production environment, specifically in fraud detection.

---

## 🛠️ Technical Workflow & Analysis

### 1. Churn Prediction Analysis (Threshold Optimization)
In this task, I evaluated a model's probability outputs against actual churn classes using a 0.5 threshold.
- **Confusion Matrix Construction:** Identified True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
- **Metric Synergy:** Balanced the trade-off between Precision (exactness) and Recall (completeness) through the F1-Score calculation.

### 2. The Accuracy Paradox in Fraud Detection
A deployed fraud detection model reported a **90.5% accuracy**, yet was deemed unsuccessful by the business unit. My analysis of the confusion matrix revealed the core issue:

| | Predicted Fraud | Predicted Normal |
|---|---|---|
| **Actual Fraud** | 5 (TP) | 5 (FN) |
| **Actual Normal** | 90 (FP) | 900 (TN) |



#### Performance Breakdown:
- **Accuracy:** 90.5% (High due to the dominance of non-fraud cases).
- **Precision:** ~5.2% (The model flags many legitimate transactions as fraud).
- **Recall:** 50% (The model misses half of the actual fraudulent activities).
- **F1-Score:** ~9.5% (Extremely low, indicating a failed model).

---

## 🚀 Key Insights & Interpretations

### What did the Data Science team overlook?
The team fell into the **Imbalanced Data** trap. In banking fraud, legitimate transactions (Majority Class) significantly outnumber fraudulent ones (Minority Class).
1. **Misleading Accuracy:** A model that predicts "Normal" for everything would still achieve high accuracy, but fail to catch any fraud.
2. **Cost of False Negatives:** Missing a fraud case (FN) is far more expensive than a false alarm (FP).
3. **Strategic Recommendation:** For production-ready fraud models, focus should shift to **Precision-Recall Curves** and **AUC-ROC** scores rather than Accuracy.

---
*Developed as part of the **Miuul Data Scientist Bootcamp** to master the evaluation of supervised learning models and the nuances of class imbalance.*
