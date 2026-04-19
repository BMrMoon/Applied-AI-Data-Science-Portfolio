##### Telco Churn Prediction #####


### Business Problem
## A machine learning model is expected to be developed that can predict customers who are likely to leave the company.

### Story of the Dataset
## The Telco customer churn dataset contains information about a fictional telecommunications company that provided home phone and internet services to 7,043 customers in California during the third quarter. It indicates which customers left, stayed, or signed up for the company’s services.

# CustomerId: Customer ID
# Gender: Gender
# SeniorCitizen: Whether the customer is a senior citizen (1, 0)
# Partner: Whether the customer has a partner (Yes, No)
# Dependents: Whether the customer has dependents (Yes, No)
# tenure: Number of months the customer has stayed with the company
# PhoneService: Whether the customer has phone service (Yes, No)
# MultipleLines: Whether the customer has multiple lines (Yes, No, No phone service)
# InternetService: Customer’s internet service provider (DSL, Fiber optic, None)
# OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)
# OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)
# DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)
# TechSupport: Whether the customer receives technical support (Yes, No, No internet service)
# StreamingTV: Whether the customer has streaming TV (Yes, No, No internet service)
# StreamingMovies: Whether the customer has streaming movies (Yes, No, No internet service)
# Contract: Customer’s contract term (Month-to-month, One year, Two years)
# PaperlessBilling: Whether the customer has paperless billing (Yes, No)
# PaymentMethod: Customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges: Monthly amount charged to the customer
# TotalCharges: Total amount charged to the customer
# Churn: Whether the customer churned (Yes or No)




# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Functions
def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool", "str"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object", "str"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    return dataframe

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

def model_scores(y_pred, y_test, y_prob=None):
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        print("AUC:", roc_auc)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Confusion Matrix:", cm)

def cv_report(model, X_train, y_train):
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print("Model: ", model)
    cv_results = cross_validate(model,
                                X_train, y_train,
                                cv=3,
                                scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

    print("Train CV Accuracy: ", cv_results['test_accuracy'].mean())
    print("Train CV Precision: ", cv_results['test_precision'].mean())
    print("Train CV Recall: ", cv_results['test_recall'].mean())
    print("Train CV F1: ", cv_results['test_f1'].mean())
    print("Train CV ROC_AUC: ", cv_results['test_roc_auc'].mean())

def main():
    df_ = pd.read_csv("./Telco-Customer-Churn.csv")
    df = df_.copy()

    df.info()
    df.describe().T
    df.isnull().sum()

    ### Task 1: Exploratory Data Analysis
    ## Step 1: Identify the numerical and categorical variables.
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    for col in df.columns:
        print(df[col].value_counts())
        print(df[col].dtype)
        print("----------------------")

    ## Step 2: Make the necessary adjustments. (Such as variables with data type errors.)
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
    df["tenure"] = df["tenure"]
    df["TotalCharges"] = df["MonthlyCharges"]*df["tenure"]

    ## Step 3: Examine the distribution of numerical and categorical variables in the dataset.
    for col in cat_cols:
        cat_summary(df, col, plot=True)
    for col in num_cols:
        num_summary(df, col, plot=True)

    ## Step 4: Examine the relationship between the categorical variables and the target variable.
    for col in cat_cols:
        target_summary_with_cat(df, "Churn", col)

    ## Step 5: Examine whether there are any outliers.
    for col in num_cols:
        print(col, check_outlier(df, col))
        target_summary_with_num(df, "Churn", col)

    ## Step 6: Examine whether there are any missing values.
    print(df.isnull().sum())
    print(df.describe().T)

    ### Görev 2 : Feature Engineering
    ## Step 1: Perform the necessary operations for missing values and outliers.
    df = replace_with_thresholds(df, "TotalCharges")

    ## Step 2: Create new variables.
    df["Demand"] = "Low"
    df.loc[(df["PhoneService"]=="Yes") & (df["MultipleLines"]=="No") & (df["SeniorCitizen"]==1), "Demand"] = "Mid"
    df.loc[(df["PhoneService"]=="Yes") & (df["MultipleLines"]=="Yes") & (df["SeniorCitizen"]==1), "Demand"] = "High"
    df["ExpectedTotalCharges"] = (df["tenure"]+1)*df["MonthlyCharges"]

    ## Step 3: Perform the encoding operations.
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]
    for col in binary_cols:
        df = label_encoder(df, col)
    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
    df = one_hot_encoder(df, ohe_cols)

    ## Step 4: Standardize the numerical variables.
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    ### Task 3: Modeling
    ## Step 1: Build models using classification algorithms, examine their accuracy scores, and select the best 4 models.
    X = df.drop(columns=['Churn', 'customerID'])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20, random_state=17)
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print("Logistic Regression")
    log_model = LogisticRegression().fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    y_prob = log_model.predict_proba(X_test)[:, 1]
    model_scores(y_prob=y_prob, y_pred=y_pred, y_test=y_test)
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print("KNN")
    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    y_prob = knn_model.predict_proba(X_test)[:, 1]
    model_scores(y_prob=y_prob, y_pred=y_pred, y_test=y_test)
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print("SVC")
    from sklearn.svm import SVC
    svc_model = SVC()
    svc_model.fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    model_scores(y_pred=y_pred, y_test=y_test)
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print("Naive Bayes")
    from sklearn.naive_bayes import GaussianNB
    nb_model = GaussianNB(priors=[sum(y_train==0)/len(y_train), (1-sum(y_train==0)/len(y_train))])
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    y_prob = nb_model.predict_proba(X_test)[:, 1]
    model_scores(y_prob=y_prob, y_pred=y_pred, y_test=y_test)
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print("Decision Tree")
    from sklearn.tree import DecisionTreeClassifier
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    y_prob = dt_model.predict_proba(X_test)[:, 1]
    model_scores(y_prob=y_prob, y_pred=y_pred, y_test=y_test)
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print("Random Forest")
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    model_scores(y_prob=y_prob, y_pred=y_pred, y_test=y_test)
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print("XGBoost")
    from xgboost import XGBClassifier
    xgboost_model = XGBClassifier()
    xgboost_model.fit(X_train, y_train)
    y_pred = xgboost_model.predict(X_test)
    y_prob = xgboost_model.predict_proba(X_test)[:, 1]
    model_scores(y_prob=y_prob, y_pred=y_pred, y_test=y_test)

    ## Step 2: Perform hyperparameter optimization for the models you selected, and rebuild the models using the hyperparameters you obtained.
    cv_report(log_model, X_train, y_train)
    cv_report(knn_model, X_train, y_train)
    cv_report(svc_model, X_train, y_train)
    cv_report(nb_model, X_train, y_train)
    cv_report(dt_model, X_train, y_train)
    cv_report(rf_model, X_train, y_train)
    cv_report(xgboost_model, X_train, y_train)

    # 1st Choice
    print(nb_model.get_params())
    probs_nb_lower = sum(y_train==0)/len(y_train)
    probs_nb_upper = 1-sum(y_train==0)/len(y_train)
    probs_list_lower = np.linspace(probs_nb_lower-0.1, probs_nb_lower+0.1, 100)
    probs_list_upper = 1 - probs_list_lower
    probs_list = [[probs_list_lower[index], probs_list_upper[index]] for index, value in enumerate(probs_list_lower)]
    nb_params = {"priors": probs_list}

    nb_gs_best = GridSearchCV(nb_model,
                               nb_params,
                               cv=5,
                               n_jobs=-1,
                               verbose=2).fit(X_train, y_train)
    print(nb_gs_best.best_params_)
    nb_final = nb_model.set_params(**nb_gs_best.best_params_).fit(X_train, y_train)
    cv_report(nb_final, X_train, y_train)


    # 2nd Choice
    print(log_model.get_params())
    log_params = {"fit_intercept": [True, False], "class_weight": [None, "balanced"]}
    log_gs_best = GridSearchCV(log_model,
                               log_params,
                               cv=5,
                               n_jobs=-1,
                               verbose=2).fit(X_train, y_train)
    print(log_gs_best.best_params_)
    log_final = log_model.set_params(**log_gs_best.best_params_).fit(X_train, y_train)
    cv_report(log_final, X_train, y_train)



if __name__ == '__main__':
    main()