# House Price Prediction: Advanced Regression Techniques 🏠📈

This project implements an end-to-end machine learning pipeline to predict residential house prices in Ames, Iowa. Using a dataset with 79 explanatory variables, I focused on advanced feature engineering, statistical imputation, and hyperparameter optimization to minimize prediction error.

---

## 🎯 Business Problem
Predicting real estate prices is a complex task due to the high number of influencing factors (location, area, quality, etc.). The goal is to develop a robust regression model that can accurately estimate the `SalePrice` of various types of houses by analyzing physical and temporal attributes.

## 📜 Dataset Story
The dataset represents the residential housing market in Ames, Iowa. It includes 79 features describing almost every aspect of residential homes.
- **Source:** Kaggle House Prices Competition.
- **Target Variable:** `SalePrice` (The property's sale price in dollars).
- **Complexity:** High dimensionality with a mix of nominal, ordinal, and continuous variables.

---

## 🛠️ Technical Workflow & Engineering Excellence

### 1. Advanced Exploratory Data Analysis (EDA)
- **Automated Classification:** Utilized a custom `grab_col_names` function to segregate categorical, numerical, and cardinal variables.
- **Missing Value Management:** Developed `na_operation` to handle complex null values. I implemented a strategic approach where variables with high missing ratios were evaluated for significance before being imputed or dropped.

### 2. Feature Engineering & Preprocessing
- **Outlier Handling:** Applied IQR-based suppression using `replace_with_thresholds` to ensure extreme values in areas or prices didn't bias the model.
- **Rare Encoding:** Implemented a rare encoder (1% threshold) to group infrequent categorical labels, preventing overfitting in high-cardinality features.
- **Derived Features:** Engineered interaction variables like `TotalArea` (combining garage, pool, and masonry veneer areas) to capture overall property magnitude.
- **Log Transformation:** Applied `log1p` transformation to the target variable to stabilize variance and normalize the distribution, followed by `expm1` for final evaluation.



### 3. Predictive Modeling & Optimization
- **Algorithms:** Focused on **Random Forest Regressor** due to its ability to handle non-linear relationships and high-dimensional data.
- **Hyperparameter Tuning:** Performed an extensive **GridSearchCV** covering `max_depth`, `max_features`, `min_samples_split`, and `n_estimators`.
- **Validation Metrics:** Evaluated performance using **RMSE (Root Mean Squared Error)** and **MAE (Mean Absolute Error)**.



---

## 🚀 Key Insights & Performance
- **Feature Importance:** Automated the ranking of features, identifying that structural quality and total living area are the primary drivers of house prices.
- **Optimization Impact:** The transition from raw data to a log-transformed, optimized Random Forest model significantly reduced the RMSE, demonstrating the power of proper statistical scaling.

---
*Developed as part of the **Miuul Data Scientist Bootcamp** to demonstrate high-level expertise in regression analysis, advanced data preprocessing, and competitive machine learning workflows.*
