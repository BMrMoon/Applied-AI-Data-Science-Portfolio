# Telco Customer Churn Prediction & Model Optimization 📞📉

This project develops an end-to-end machine learning pipeline to predict customer churn for a telecommunications company. By implementing advanced feature engineering and comparing multiple classification algorithms, I achieved a robust predictive model to help the business minimize customer attrition.

---

## 🎯 Business Problem
The objective is to identify customers who are likely to leave the company (Churn). Early detection allows the business to implement retention strategies, such as personalized offers or improved tech support, ultimately protecting the company's revenue.

## 📜 Dataset & Business Context
The dataset provides information on 7,043 customers in California. It includes demographic data, services they signed up for, and account information.
- **Key Indicators:** Tenure (loyalty), Contract Type, Internet Service (Fiber optic vs DSL), and Monthly Charges.
- **Target Variable:** `Churn` (Yes: 1, No: 0).

---

## 🛠️ Technical Workflow & Feature Engineering

### 1. Exploratory Data Analysis (EDA)
- **Variable Classification:** Developed a custom `grab_col_names` function to automatically categorize categorical, numerical, and cardinal variables.
- **Target Insights:** Analyzed the impact of contract types and internet services on churn rates, identifying Fiber optic users as a high-risk group.

### 2. Advanced Feature Engineering
I moved beyond the raw data to create domain-specific features:
- **Demand Segmentation:** Created a `Demand` variable by cross-referencing `SeniorCitizen` status with `MultipleLines` and `PhoneService` usage.
- **Expected Total Charges:** Engineered a predictive billing variable based on tenure projections to capture spending patterns.
- **Robust Preprocessing:** Handled outliers in `TotalCharges` and implemented a combination of **Label Encoding** and **One-Hot Encoding** for categorical data.

### 3. Model Selection & Hyperparameter Tuning
I evaluated and compared seven different classification algorithms:
- **Algorithms:** Logistic Regression, KNN, SVC, Naive Bayes, Decision Tree, Random Forest, and XGBoost.
- **Optimization:** Performed **GridSearchCV** on the top-performing models. 
    - *Naive Bayes:* Optimized class priors through an automated linspace search.
    - *Logistic Regression:* Tuned intercept and class weights to handle data balance issues.
- **Evaluation:** Used 5-fold Cross-Validation with focus on **ROC_AUC, Precision, and Recall** to ensure the model generalizes well.



---

## 🚀 Key Insights & Performance
- **Model Efficiency:** Demonstrated that simpler models like **Logistic Regression** and **Gaussian Naive Bayes** can achieve high performance when backed by strong feature engineering.
- **Business Impact:** The final model identifies high-risk customers with a high ROC_AUC score, enabling the marketing team to focus their resources on the most critical 20% of the customer base.

---
*Developed as part of the **Miuul Data Scientist Bootcamp** to demonstrate high-level predictive analytics, statistical modeling, and automated hyperparameter optimization.*
