# CLTV Prediction & SQL Data Insights 📊🗄️

This project combines advanced **Predictive Analytics** with robust **SQL Data Engineering** to provide a 360-degree view of customer behavior for **FLO**. By integrating BG-NBD/Gamma-Gamma models with complex SQL queries, I have established a framework for both historical reporting and future value prediction.

---

## 🎯 Business Problem & Data Story
FLO seeks to optimize its medium-to-long-term marketing strategies. The analysis is based on omnichannel shopping data (2020-2021), focusing on identifying high-value customers and understanding purchase patterns across different platforms to facilitate data-driven decision-making.

---

## 🗄️ SQL Data Engineering & Insights
Beyond predictive modeling, I performed extensive data exploration using **PostgreSQL** to extract key business metrics directly from the database.

### Key SQL Implementations:
* **Advanced Data Manipulation:** Used `LATERAL unnest` and `STRING_TO_ARRAY` to parse and analyze bracketed, comma-separated category data within the customer database.
* **Window Functions:** Implemented `ROW_NUMBER()` over `PARTITION BY` to identify the most active customers and most popular categories for each shopping channel.
* **Omnichannel Analytics:** Developed queries to calculate Total Revenue (Offline + Online), Average Order Value (AOV), and purchase frequency metrics.
* **Temporal Breakdown:** Created complex `CASE WHEN` logic combined with `EXTRACT` functions to analyze customer tenure and historical trends.

---

## 🛠️ Predictive Modeling (BG-NBD & Gamma-Gamma)
To complement the historical SQL analysis, I implemented a probabilistic framework in Python:
1. **BG-NBD Model:** Forecasted expected transaction counts for 3-month and 6-month horizons to understand purchase frequency trends.
2. **Gamma-Gamma Model:** Estimated the expected average profit (monetary value) per customer.
3. **CLTV Segmentation:** Segmented the entire database into 4 tiers (A, B, C, D) based on 6-month projected lifetime value.



---

## 🚀 Strategic Results
- **Dynamic Decision Making:** Identified the "Champions" (Segment A) for exclusive loyalty programs and "At Risk" segments for reactivation through SQL-based frequency analysis.
- **Scalable Reporting:** The SQL scripts provide a production-ready way for marketing teams to pull real-time insights on category popularity and channel-specific performance.

---
*Developed during the **Miuul Data Scientist Bootcamp** to demonstrate the seamless integration of SQL database management with advanced Machine Learning workflows.*
