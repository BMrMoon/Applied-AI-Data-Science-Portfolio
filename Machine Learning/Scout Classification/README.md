# Football Talent Scouting Classification with Machine Learning ⚽📊

This project develops an advanced machine learning pipeline to automate the **Talent Scouting** process. By analyzing scout ratings for football players, the system classifies players into "Average" or "Highlighted" categories, enabling data-driven recruitment and potential evaluation.

---

## 🎯 Business Problem
The objective is to predict the potential class of football players based on the attributes and scores given by scouts. This automation helps scouting departments prioritize high-potential players and minimizes human bias in the initial evaluation phase.

## 📜 Dataset Story
The data consists of attribute-based evaluations from **Scoutium**. It includes ratings for various physical and technical skills of football players across different matches and positions.
- **Attributes:** Scores for specific skills (e.g., passing, speed, positioning).
- **Labels:** Final decisions by scouts (`average`, `highlighted`).
- **Scope:** Excludes goalkeepers and "below_average" outliers to focus on outfield talent prediction.

---

## 🛠️ Technical Workflow & Pipeline Architecture

### 1. Data Engineering & Transformation
- **Pivot Transformation:** Converted raw evaluation logs into a player-based matrix where each row represents a unique player profile with their corresponding attribute scores.
- **Categorical Processing:** Handled player positions and encoded the target potential labels using **Label Encoding**.
- **Standardization:** Applied **StandardScaler** to numerical attributes to ensure distance-based and gradient-based algorithms perform optimally.

### 2. Automated ML Pipeline
I developed a modular `ML_pipeline` that handles the entire model lifecycle:
- **Comprehensive Benchmarking:** Simultaneously trains and evaluates 10 different algorithms:
  - *Tree-based:* Decision Tree, Random Forest, GBM, XGBoost, LightGBM, AdaBoost.
  - *Linear & Statistical:* Logistic Regression, SVM, Gaussian Naive Bayes.
  - *Distance-based:* K-Nearest Neighbors (KNN).
- **Hyperparameter Optimization:** Conducted automated **GridSearchCV** for each model, tuning critical parameters like `n_estimators`, `max_depth`, and `learning_rate`.

### 3. Ensemble Learning (Voting Classifier)
To maximize predictive power and stability:
- **Model Selection:** The pipeline automatically identifies models performing above the mean F1-weighted score.
- **Soft Voting:** Selected models are combined into a **Voting Classifier**, which aggregates predicted probabilities to deliver a final, more robust classification.



---

## 🚀 Performance Metrics & Evaluation
- **Evaluation Strategy:** Used **3-fold Cross-Validation** during training to prevent overfitting.
- **Metrics:** Evaluated models based on **Accuracy, Precision, Recall, and F1-Score**.
- **Result:** The ensemble approach significantly outperformed individual models, providing a highly reliable "Highlighted" player identification system.

---
*Developed as part of the **Scoutium Case Study** to demonstrate expertise in building automated ML pipelines, hyperparameter tuning, and advanced ensemble learning techniques.*
