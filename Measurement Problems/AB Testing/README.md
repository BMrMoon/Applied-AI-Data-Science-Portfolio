# A/B Test Comparison of Bidding Methods 📊🎯

This project implements a rigorous statistical A/B test to evaluate the performance of Facebook's "Average Bidding" method against the established "Maximum Bidding" strategy for an e-commerce platform.

---

## 🎯 Business Problem
The client, **bombabomba.com**, wants to determine if the newly introduced "Average Bidding" type generates more conversions (purchases) than the current "Maximum Bidding" method. 
- **Success Metric:** The primary KPI for this analysis is **Purchase**.
- **Objective:** Statistically prove whether the difference in average purchases between the two bidding methods is significant or occurred by chance.

## 📜 Dataset & Business Context
The dataset contains a month's worth of ad interaction data divided into two groups:
- **Control Group (Maximum Bidding):** Traditional bidding method.
- **Test Group (Average Bidding):** New alternative bidding method.

### Key Metrics:
* **Impression:** Number of ad views.
* **Click:** Number of clicks on displayed ads.
* **Purchase:** Number of products purchased after clicks (Target Metric).
* **Earning:** Total revenue generated.

---

## 🛠️ Technical Workflow & Statistical Rigor

### 1. Data Exploration & Pipeline
- Conducted descriptive analysis for both control and test groups.
- Developed a dynamic data preprocessing pipeline to handle multiple Excel sheets and group labeling.

### 2. Hypothesis Testing Framework
I implemented a comprehensive statistical decision tree to ensure the validity of the results:
- **Normality Assumption:** Applied the **Shapiro-Wilk Test** to check if the purchase data follows a normal distribution.
- **Variance Homogeneity:** Applied **Levene’s Test** to verify if the groups have equal variances.
- **Test Selection:**
    - If assumptions were met: **Independent Two-Sample T-Test** (Parametric).
    - If variances were unequal: **Welch’s T-Test**.
    - If normality was violated: **Mann-Whitney U Test** (Non-parametric).

### 3. Automated Decision Logic
- Functionalized the entire process from data ingestion to test selection and result interpretation (`select_test` and `test_result` functions), ensuring a reproducible and scalable testing framework.

---

## 🚀 Key Strategic Results
- **Statistical Significance:** Interpreted the p-values to determine if the null hypothesis ($H_0$: No significant difference) could be rejected.
- **Business Guidance:** Provided a clear recommendation to the client on whether to switch to Average Bidding based on mathematical evidence rather than raw average comparisons.

---
*Developed during the **Miuul Data Scientist Bootcamp** to demonstrate my ability to apply scientific methods to digital marketing and conversion optimization.*
