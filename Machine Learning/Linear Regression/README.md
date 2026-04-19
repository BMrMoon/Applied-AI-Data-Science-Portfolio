# Error Evaluation for Regression Models: Salary Prediction 📉💰

This project demonstrates the fundamental mathematical evaluation of a Linear Regression model. By manually implementing core error metrics, I analyzed the performance of a salary prediction model based on years of experience.

---

## 🎯 Project Objective
The goal is to evaluate a predefined linear regression model by calculating its error margins. This allows for a granular understanding of how specific metrics like MSE, RMSE, and MAE penalize deviations from actual data points.

## 📜 Business Scenario
We are predicting salaries based on years of experience using a model with a fixed bias ($b=275$) and weight ($w=90$).
- **Model Equation:** $\hat{y} = 275 + 90x$
- **Target:** Compare predicted salaries ($\hat{y}$) against actual salaries ($y$) to quantify model performance.

---

## 🛠️ Technical Workflow & Mathematical Rigor

### 1. Model Implementation
- Developed a custom linear regression function to calculate the `expected_salary` for a dataset of 15 observations.
- Computed the residuals (errors) for each data point to observe individual deviations.

### 2. Performance Metrics
I implemented the following statistical error evaluation methods:
- **MSE (Mean Squared Error):** Measures the average of the squares of the errors. It heavily penalizes large errors, making it sensitive to outliers.
- **RMSE (Root Mean Squared Error):** The square root of MSE, bringing the error metric back to the same unit as the target variable (Salary) for better interpretability.
- **MAE (Mean Absolute Error):** The average of the absolute differences between predictions and actual values, providing a robust and linear representation of error.

---

## 🚀 Key Insights
- **Manual Computation:** By calculating these metrics without high-level libraries, I demonstrated a deep understanding of the mathematical foundation behind model evaluation.
- **Error Interpretation:** Comparing MAE and RMSE helps identify whether the model's performance is being skewed by specific outliers or if the error is distributed consistently.

---
*Developed as part of the **Miuul Data Scientist Bootcamp** to master the mathematical foundations of regression analysis and predictive model validation.*
