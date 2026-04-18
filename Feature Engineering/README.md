# Diabetes Prediction & Advanced Feature Engineering 🩺📉

This project demonstrates a comprehensive **Feature Engineering** and **Exploratory Data Analysis (EDA)** pipeline to predict diabetes using the Pima Indians Diabetes dataset. The core focus is on handling implicit missing values, sophisticated data imputation, and the generation of interaction features to enhance model performance.

---

## 🎯 Business Problem
The objective is to build a machine learning model that predicts whether a patient has diabetes based on diagnostic measurements. This project highlights that a model's predictive power is deeply rooted in the quality of data preprocessing—specifically how zero-values in biological indicators are treated and how new variables are engineered to capture complex relationships.

## 📜 Dataset Story
The dataset is provided by the **National Institute of Diabetes and Digestive and Kidney Diseases**. It focuses on Pima Indian women aged 21 and older living in Phoenix, Arizona.

### Key Features:
* **Glucose / Insulin:** 2-hour plasma glucose concentration and serum insulin.
* **BMI / Blood Pressure:** Body mass index and diastolic blood pressure.
* **Outcome:** Target variable (1: Positive for diabetes, 0: Negative).
* **DiabetesPedigreeFunction:** A score representing the likelihood of diabetes based on family history.

---

## 🛠️ Technical Workflow & Engineering Excellence

### 1. Robust Missing Value Strategy
Recognizing that biological indicators like **Glucose, Insulin, BMI, and Blood Pressure** cannot realistically be zero in a living patient, I treated these "0" values as implicit missing data (**NaN**).
- **Visualization:** Used `missingno` (MSNO) to analyze the sparsity and correlation of missingness across variables.
- **K-Nearest Neighbors (KNN) Imputation:** Instead of simple mean/median filling, I implemented `KNNImputer` to estimate missing values based on similar patient profiles, preserving the multidimensional structure of the data.

### 2. Strategic Feature Engineering
I developed new interaction features to capture non-linear biological relationships:
- **Metabolic Indicators:** Derived variables like `Glucose / Insulin` ratio to represent metabolic efficiency.
- **Risk Multipliers:** Engineered interaction terms such as `Age * Glucose` and `Pregnancies * BMI` to capture intensified risk zones in specific demographic clusters.

### 3. Pipeline & Statistical Rigor
- **Outlier Management:** Applied IQR-based thresholds (1% - 99%) to handle extreme values while maintaining overall distribution integrity.
- **Standardization:** Utilized `RobustScaler` to scale numerical variables, ensuring the model remains resilient to any remaining outliers.
- **Feature Importance:** Automated the ranking of features to visualize which indicators (e.g., Glucose levels) drive the model's predictive power.

---

## 🚀 Key Results
- **Model Choice:** Employed a `RandomForestClassifier` to handle the complexity and non-linear patterns within the health data.
- **Performance:** Successfully improved the model's baseline performance through targeted imputation and high-impact feature interaction terms.

---
*Developed during the **Miuul Data Scientist Bootcamp** to showcase the critical role of Feature Engineering and data-driven insights in healthcare analytics.*
