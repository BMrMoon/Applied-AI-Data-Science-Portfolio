# Rating Product & Sorting Reviews in Amazon 🛒⭐

This project addresses two critical challenges in e-commerce: accurately calculating product ratings and ranking user reviews fairly. By implementing **Time-Based Weighted Averages** and **Wilson Lower Bound** scores, we enhance customer satisfaction and prevent misleading content from dominating product pages.

---

## 🎯 Business Problem
In e-commerce, misleading ratings and improperly ranked reviews can directly impact sales, leading to financial loss and customer dissatisfaction. The goals of this project are:
- **Dynamic Rating:** Calculating a product rating that reflects recent customer trends and quality shifts.
- **Fair Ranking:** Ensuring that the most helpful and statistically reliable reviews appear at the top of the product page.

## 📜 Dataset & Business Context
The dataset contains Amazon electronics category metadata, including user ratings and review helpfulness scores.

### Key Features:
* **overall:** Product rating (1-5).
* **day_diff:** Number of days since the review was posted.
* **helpful_yes:** Number of times a review was found helpful by other users.
* **total_vote:** Total number of votes (Helpful + Not Helpful) given to a review.

---

## 🛠️ Technical Workflow & Core Competencies

### 1. Advanced Product Rating
Implemented a **Time-Based Weighted Average** to prioritize recent customer feedback. This ensures the rating reflects the current state of the product rather than relying solely on outdated historical data.
- **Weights Applied:** Recent reviews (<30 days) are weighted significantly higher (24%) compared to older reviews (>360 days, 16%).

### 2. Review Ranking Logic
Moving beyond simple "upvote minus downvote" logic, I implemented three different scoring methods to identify the most helpful reviews:
- **Score Positive-Negative Difference:** Basic subtraction of votes.
- **Average Rating Score:** Ratio of helpful votes to total votes.
- **Wilson Lower Bound (WLB) Score:** A statistically robust method that calculates the lower bound of the confidence interval for a Bernoulli parameter. This prevents reviews with very few votes from unfairly dominating the top rankings.

### 3. Data Engineering
- Functionalized preprocessing steps including missing value checks and descriptive statistics.
- Engineered the `helpful_no` variable to quantify negative feedback accurately for statistical modeling.

---

## 🚀 Key Strategic Results
- **Dynamic Response:** Developed a rating system that responds faster to recent product improvements or seasonal quality changes.
- **Statistical Reliability:** By using **Wilson Lower Bound**, the system ensures that reviews with a high volume of interaction and proven helpfulness are prioritized over those with high ratios but low sample sizes.

---
*Developed during the **Miuul Data Scientist Bootcamp** to apply my MSc-level analytical and statistical skills to complex e-commerce optimization problems.*
