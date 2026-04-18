# Imports
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve


# Functions 
def main():
    # Task 1
    df = pd.DataFrame()
    df["classes"] = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    df["probabilities"] = np.array([0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25])
    df["predictions"] = df["probabilities"].apply(lambda x: 1 if x > 0.5 else 0)

    cm = confusion_matrix(df["classes"], df["predictions"])

    TP = cm[0][0]
    FP = cm[1][0]
    TN = cm[1][1]
    FN = cm[0][1]

    accuracy = accuracy_score(df["classes"], df["predictions"])
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (recall * precision) / (recall + precision)

    # Task 2
    TP = 5
    TN = 900
    FP = 90
    FN = 5

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (recall * precision) / (recall + precision)



if __name__ == '__main__':
    main()