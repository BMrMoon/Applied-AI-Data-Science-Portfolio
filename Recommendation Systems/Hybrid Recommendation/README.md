# Hybrid Recommender System: MovieLens 🎬🍿

This project implements a comprehensive recommendation engine that synergizes **User-Based** and **Item-Based** collaborative filtering techniques. By leveraging the MovieLens dataset (20 million ratings), the system generates 10 personalized movie recommendations (5 from each method) for a target user.

---

## 🎯 Business Problem
The goal is to provide highly personalized movie suggestions by utilizing two distinct perspectives:
1. **User-Based:** Identifying users with similar viewing patterns and recommending movies they highly rated.
2. **Item-Based:** Recommending movies that are statistically similar to the ones the user recently enjoyed and rated with 5 stars.

## 📜 Dataset & Business Context
The dataset is provided by **MovieLens**, a movie recommendation service. It contains approximately **20 million ratings** for **27,000+ movies** by **138,000+ users**.

### Data Sources:
* **movie.csv:** Metadata containing `movieId`, `title`, and `genres`.
* **rating.csv:** Behavioral data containing `userId`, `movieId`, `rating` (1-5), and `timestamp`.

---

## 🛠️ Technical Workflow

### 1. Data Preparation & Robustness
- **Threshold Filtering:** To ensure statistical reliability and avoid "cold start" noise, movies with fewer than 1,000 ratings were excluded from the analysis.
- **Pivot Transformation:** Created a sparse matrix where indices are users and columns are movie titles, facilitating high-dimensional correlation analysis.

### 2. User-Based Collaborative Filtering
- **Cohort Identification:** Selected a cohort of users who watched at least 60% of the same movies as the target user.
- **Correlation Mapping:** Calculated Pearson Correlation between the target user and the cohort.
- **Weighted Average Score:** Instead of simple averages, I calculated a `weighted_rating` by multiplying the peer's rating by their correlation score. This ensures that users with the most similar tastes have the highest impact on recommendations.

### 3. Item-Based Collaborative Filtering
- **Recent Preference Tracking:** Identified the target user's most recently watched movie rated with 5.0 stars.
- **Statistical Similarity:** Used `corrwith()` to compute the correlation between that specific film and all other movies in the filtered pivot table, selecting the top 5 most similar titles.



[Image of user-based vs item-based collaborative filtering]


---

## 🚀 Key Strategic Results
- **Dual Perspective:** The hybrid approach balances "social proof" (what people like you watch) with "content consistency" (what is similar to your favorites).
- **Scalable Logic:** The functions are designed to handle large-scale pivot tables efficiently, allowing for updates as new rating data becomes available.

---
*Developed during the **Miuul Data Scientist Bootcamp** to demonstrate advanced expertise in collaborative filtering and recommendation pipeline architecture.*
