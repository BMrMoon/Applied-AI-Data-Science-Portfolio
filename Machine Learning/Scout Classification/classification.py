# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


# Functions
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def ensemble_classifier(models, X_train, y_train, X_test, y_test, cv=5, scoring=["accuracy", "f1", "precision", "recall"]):
    voting_clf = VotingClassifier(estimators=models,
                                  voting='soft').fit(X_train, y_train)
    cv_results = cross_validate(voting_clf, X_train, y_train, cv=cv, scoring=scoring)
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"Precision: {cv_results['test_precision'].mean()}")
    print(f"Recall: {cv_results['test_recall'].mean()}")
    y_pred = voting_clf.predict(X_test)
    test_report = classification_report(y_test, y_pred)
    print(f"Test Classification Report:\n {test_report}", end="\n\n")
    return voting_clf

def ML_pipeline(X, y, scoring='f1_weighted', cv=3):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20, random_state=17, shuffle=True)

    classifiers = {
        "RTC": DecisionTreeClassifier(),
        "RFC": RandomForestClassifier(),
        "LR": LogisticRegression(),
        "SVM": SVC(probability=True),
        "LGBM": LGBMClassifier(),
        "XGBoost": XGBClassifier(),
        "NB": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "GBM": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
    }

    nb_priors_lower = np.linspace(0.01, 0.99, 100)
    nb_priors_upper = 1 - nb_priors_lower
    nb_priors = [[nb_priors_lower[index], nb_priors_upper[index]] for index in range(len(nb_priors_upper))]
    params = {
        "RTC": {
            'max_depth': list(range(1, 20)),
            "min_samples_split": list(range(2, 30))},
        "RFC": {
            "max_depth": [8, 15, None],
            "max_features": [3, 5, 7],
            "min_samples_split": [15, 20],
            "n_estimators": [200, 300]},
        "LR": {
            "fit_intercept": [True, False],
            "class_weight": [None, "balanced"]},
        "SVM": {
            "C": [0.1, 1, 10, 50],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
            "probability": [True]},
        "LGBM": {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [300, 500],
            "colsample_bytree": [0.7, 1]},
        "XGBoost": {
            "learning_rate": [0.1, 0.01],
            "max_depth": [5, 8],
            "n_estimators": [100, 200],
            "colsample_bytree": [0.5, 1]},
        "NB":{
            "priors": nb_priors},
        "KNN": {
            "n_neighbors": range(2, 50)},
        "GBM": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.2, 0.1, 0.01],
            "max_depth": [3, 5, 7]},
        "AdaBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.2, 0.1, 0.01]
        },
    }

    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier in classifiers.items():
        best_models[name] = {}
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params[name], cv=cv, n_jobs=-1, verbose=3).fit(X_train, y_train)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        print("Testing is in progress...")
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(f"Classification Report:\n {report}", end="\n\n")

        best_models[name]["model"] = final_model
        best_models[name]["f1_weighted"] = cv_results['test_score'].mean()

    df_models = {"model": [], "f1_weighted": []}
    for name, infos in best_models.items():
        df_models["model"].append(name)
        df_models["f1_weighted"].append(infos["f1_weighted"])
    df_models = pd.DataFrame(df_models)
    models_selected = df_models[df_models["f1_weighted"] >= df_models["f1_weighted"].mean()]["model"]
    models = []
    for model in models_selected:
        models.append((model, best_models[model]["model"]))
    voting_clf = ensemble_classifier(models, X_train, y_train, X_test, y_test, cv=cv)

    return best_models, voting_clf

def plot_importance(model, features, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:len(features)])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def main():
    ## Step 1: Read the scoutium_attributes.csv and scoutium_potential_labels.csv files.
    df_attributes = pd.read_csv('./scoutium_attributes.csv', sep=';')
    df_potential_labels = pd.read_csv('./scoutium_potential_labels.csv', sep=';')

    ## Step 2: Merge the CSV files you have read using the merge function. (Perform the merge operation using the four variables: "task_response_id", "match_id", "evaluator_id", and "player_id".)
    df_ = df_attributes.merge(df_potential_labels, on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'])
    df = df_.copy()

    ## Step 3: Remove the Goalkeeper (1) class from the position_id variable in the dataset.
    df = df.drop(index=df[df["position_id"]==1].index, axis=0)

    ## Step 4: Remove the below_average class from the potential_label variable in the dataset. (below_average constitutes 1% of the entire dataset.)
    df = df.drop(index=df[df["potential_label"]=="below_average"].index, axis=0)

    ## Step 5: Using the dataset you created, build a table with the pivot_table function. In this pivot table, perform the necessary transformation so that each row represents a single player.
    # Step 5.1: Create the pivot table such that the index consists of player_id, position_id, and potential_label, the columns consist of attribute_id, and the values consist of attribute_value, which are the scores given by scouts to the players.
    df.pivot_table(index=["player_id", "position_id", "potential_label"], columns="attribute_id", values="attribute_value")

    # Step 5.2: Use the reset_index function to assign the index values as variables, and convert the names of the attribute_id columns to strings.
    df.reset_index(drop=False, inplace=True)

    ## Step 6: Use the Label Encoder function to represent the potential_label categories (average, highlighted) numerically.
    df = label_encoder(dataframe=df, binary_col='potential_label')

    ## Step 7: Assign the numerical variable columns to a list named num_cols.
    num_cols = df.select_dtypes(include="number").columns
    num_cols = [col for col in num_cols if df[col].nunique()>2]

    ## Step 8: Apply StandardScaler to scale the data in all variables stored in num_cols.
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    ## Step 9: Using the dataset at hand, develop a machine learning model that predicts football players’ potential labels with minimum error. (Print the roc_auc, f1, precision, recall, and accuracy metrics.)
    X = df.drop(columns=["potential_label"])
    y = df["potential_label"]
    bests, clf = ML_pipeline(X=X, y=y, cv=2)



if __name__ == '__main__':
    main()