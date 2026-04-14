# Customer Segmentation via RFM Analysis 📊

### Applied CRM Analytics & Data Science Case Study
I am an Electrical and Electronics Engineer (MSc) driven by a passion for transforming complex datasets into actionable business insights. This project, developed during the **Miuul Data Scientist Bootcamp**, focuses on behavioral clustering and strategic marketing.

---

## 🎯 Business Problem
**FLO**, an omnichannel shoe retailer, seeks to segment its customers to develop tailored marketing strategies. By identifying distinct behavioral patterns, we aim to optimize customer engagement and maximize lifetime value.

## 📜 Dataset & Business Context
The analysis is based on the shopping history of customers who made purchases through **OmniChannel** (Online & Offline) platforms between **2020–2021**.

### Core Variables:
* **master_id:** Unique customer identifier.
* **order_channel:** Platforms used (Android, iOS, Desktop, Mobile, Offline).
* **order_num_total:** Aggregate number of purchases across all channels.
* **customer_value_total:** Total monetary spending.
* **interested_in_categories_12:** Customer interest profiles from the last 12 months.

---

## 🛠️ Technical Workflow & Core Competencies

### 1. Data Engineering & Preprocessing
* **Outlier Management:** Implemented robust suppression using IQR thresholds to maintain data integrity.
* **Feature Engineering:** Merged online and offline metrics to create a unified "Omnichannel" view of the customer.
* **Type Conversion:** Standardized date-time variables for accurate recency calculations.

### 2. RFM Analytics & Scoring
* **Recency:** Calculated days since the last purchase to measure engagement.
* **Frequency:** Quantified total purchase counts to determine loyalty.
* **Monetary:** Measured total spending to identify high-value segments.
* **Scoring:** Applied `pd.qcut` to normalize metrics into 1-5 scores.

### 3. Segmentation Strategy
Used **Regex-based mapping** to categorize the customer base into 10 actionable segments, including *Champions, Loyal Customers, At Risk, and Hibernating*.

---

## 🚀 Strategic Business Insights
The model directly supports high-impact business decisions:
- **Premium Targeting:** Identified *Champions* and *Loyal Customers* for an exclusive women’s shoe brand launch.
- **Retention Engineering:** Automated the identification of *At Risk* and *Hibernating* segments for a 40% discount campaign in Men’s and Children’s categories.

---
*Developed as part of the Miuul Data Scientist Bootcamp.*
