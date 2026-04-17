# Association Rule Based Recommender System 🛠️🛒

This project implements a recommendation engine for **Armut**, Turkey's largest online service platform. Using **Association Rule Learning (ARL)**, the system analyzes historical service purchase patterns to suggest complementary services to users, similar to "frequently bought together" features.

---

## 🎯 Business Problem
Armut brings together service providers and customers for various needs like cleaning, renovation, and moving. The objective is to increase service cross-selling and user engagement by building a recommendation system based on the services customers have previously purchased.

## 📜 Dataset & Business Context
The dataset contains information about customers, the anonymized services they received, their categories, and purchase timestamps.

### Key Features:
* **UserId:** Unique identifier for the customer.
* **ServiceId:** Anonymized services (unique within each category).
* **CategoryId:** Anonymized service categories (e.g., cleaning, moving).
* **CreateDate:** The date and time the service was purchased.

---

## 🛠️ Technical Workflow & Algorithms

### 1. Data Engineering & Basket Definition
- **Feature Engineering:** Combined `ServiceId` and `CategoryId` to create a unique identifier for each specific service (e.g., `2_0`).
- **Basket Logic:** Since services don't have an invoice ID, I defined a "basket" as the collection of services a user receives within a specific month. A unique `SepetID` was created by merging `UserId` and the `Year-Month` string.
- **Pivot Transformation:** Transformed the data into a binary (Boolean) matrix where rows represent baskets and columns represent services.

### 2. Association Rule Learning (Apriori)
I utilized the **Apriori Algorithm** to identify frequent itemsets and generate association rules based on three key metrics:
- **Support:** How frequently the service combination appears in the entire dataset.
- **Confidence:** The probability that a customer who receives Service A will also receive Service B.
- **Lift:** How much the purchase of Service A increases the probability of purchasing Service B, accounting for the popularity of both.



### 3. Recommendation Engine
- **`arl_recommender`:** Developed a custom function that filters and sorts association rules by **Lift** to provide the most relevant top-N recommendations for any given service.

---

## 🚀 Key Results
- **Actionable Insights:** Successfully generated rules that predict, for example, which renovation services are likely to be requested after a cleaning service.
- **Scalability:** The logic is designed to be retrained periodically to adapt to changing user behavior and seasonal service trends.

---
*Developed during the **Miuul Data Scientist Bootcamp** to apply unsupervised learning techniques to real-world recommendation problems.*
