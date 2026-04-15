# Customer Lifetime Value Prediction via BG-NBD & Gamma-Gamma 📈

This project focuses on establishing a strategic roadmap for **FLO**, an omnichannel shoe retailer, by estimating the potential future value of its existing customer base. Using probabilistic models, we project transaction counts and average profitability to calculate **Customer Lifetime Value (CLTV)** for mid-to-long-term planning.

---

## 🎯 Business Problem
To facilitate data-driven decision-making, FLO needs to predict the potential value that existing customers will generate in the future. This roadmap is essential for:
- Optimizing marketing budget allocation.
- Developing personalized customer retention strategies.
- Identifying high-value customer segments for priority engagement.

## 📜 Dataset & Business Context
The analysis is based on information derived from the past shopping behaviors of customers who made purchases through **OmniChannel** (both online and offline) platforms during **2020–2021**.

### Key Features:
* **master_id:** Unique customer identifier.
* **order_num_total:** Aggregate number of purchases (Online + Offline).
* **customer_value_total:** Total monetary spending across online and offline channels.
* **first_order_date / last_order_date:** Temporal data used to calculate recency and tenure.

---

## 🛠️ Technical Workflow & Core Competencies

### 1. Data Engineering & Preprocessing
* **Outlier Handling:** Implemented IQR-based thresholds (1% - 99%) to suppress outliers in purchase counts and monetary values, ensuring the stability of the probabilistic models.
* **Feature Engineering:** Merged offline and online transaction data to create a unified view of customer behavior.
* **Metric Standardization:** Converted temporal data into weekly units (`recency_cltv_weekly`, `T_weekly`) as required by the models.

### 2. Probabilistic Modeling (BG-NBD & Gamma-Gamma)
* **BG-NBD Model (Beta Geometric / Negative Binomial Distribution):** Used to forecast the expected number of transactions. Predicted expected purchases for **3-month** and **6-month** horizons.
* **Gamma-Gamma Model:** Utilized to estimate the expected average profit (monetary value) per customer.
* **CLTV Integration:** Combined both models to calculate the **6-month projected CLTV** with a 1% monthly discount rate.

### 3. Segmentation Strategy
* Divided the customer base into 4 distinct segments (**A, B, C, D**) based on their 6-month CLTV scores using quartile-based grouping.

---

## 🚀 Strategic Business Insights
The model provides actionable recommendations for FLO management:
- **Segment A (High-Value):** Top-tier customers with the highest expected profitability. *Recommendation:* Exclusive loyalty rewards and early access to new collections.
- **Segment C & D (At Risk/Low Value):** Customers with declining engagement or low frequency. *Recommendation:* Reactivation campaigns and cost-effective automated reminders.

---
*Developed during the **Miuul Data Scientist Bootcamp** to apply my MSc-level analytical skills to real-world predictive analytics problems.*
