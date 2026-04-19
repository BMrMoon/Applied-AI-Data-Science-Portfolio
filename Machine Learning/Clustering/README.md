# Customer Segmentation with Unsupervised Learning: FLO Case Study 🛍️🤖

This project implements advanced **Unsupervised Learning** techniques to segment FLO's customer base into actionable behavioral clusters. By leveraging both **K-Means** and **Hierarchical Clustering** algorithms, I have developed a multi-layered segmentation strategy to optimize targeted marketing activities.

---

## 🎯 Business Problem
FLO seeks to understand its diverse customer base by identifying distinct behavioral patterns. The objective is to group customers with similar traits to develop specialized marketing strategies, improve customer retention, and maximize lifetime value through cluster-based insights.

## 📜 Dataset & Business Context
The dataset contains historical omnichannel (online and offline) purchase data from 2020–2021.
- **Key Indicators:** Order frequency, total spending, recency, and customer tenure.
- **Goal:** Grouping over 20,000 customers based on purchase behavior rather than pre-defined labels.

---

## 🛠️ Technical Workflow & Algorithmic Excellence

### 1. Feature Engineering & Preprocessing
- **Metric Generation:** Engineered new behavioral features including `Tenure` (customer age in days), `Recency`, and `Daily Average Spent` to capture the velocity of customer interactions.
- **Robust Scaling:** Implemented **RobustScaler** to standardize numerical variables, ensuring the distance-based clustering algorithms are not biased by outliers.
- **Encoding:** Applied Label Encoding and One-Hot Encoding for categorical platform and channel data.

### 2. K-Means Clustering
- **The Elbow Method:** Utilized **KElbowVisualizer** and Sum of Squared Errors (SSE) analysis to mathematically determine the optimal number of clusters.
- **Centroid Analysis:** Built the final K-Means model to partition the customer base into distinct, non-overlapping segments.



### 3. Hierarchical Clustering (HC)
- **Agglomerative Approach:** Implemented hierarchical clustering using the "average" linkage method.
- **Dendrogram Analysis:** Visualized the tree-like structure of customer relationships to identify natural nesting and validate the cluster consistency found in K-Means.



---

## 🚀 Strategic Results & Insights
- **Segment Profiling:** Conducted extensive statistical analysis on each cluster to define their core traits (e.g., "High Spenders with Low Frequency" vs. "Loyal Everyday Shoppers").
- **Marketing Integration:** Developed a dual-validation framework where K-Means results were cross-checked with Hierarchical Clustering to ensure robust and stable segment definitions.
- **Actionable KPIs:** Provided mean and median statistics for each cluster, allowing the marketing team to set data-driven targets for each customer group.

---
*Developed as part of the **Miuul Data Scientist Bootcamp** to demonstrate high-level expertise in unsupervised machine learning, statistical data exploration, and customer behavioral modeling.*
