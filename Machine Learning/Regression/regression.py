
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation


# Imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 500)
np.set_printoptions(threshold=sys.maxsize)


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

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object", "str"]]

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

def df_column_check(dataframe):
    column_name = []
    n_element = []
    type = []
    for col in dataframe.columns:
        column_name.append(col)
        n_element.append(len(dataframe[col].unique()))
        type.append(dataframe[col].dtype)
    return pd.DataFrame({"column_name": column_name, "n_element": n_element, "type": type})

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

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

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
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit.astype(dataframe[variable].dtype)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit.astype(dataframe[variable].dtype)
    return dataframe

def missing_values_table(dataframe, drop_thr, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    type = dataframe[na_columns].dtypes
    missing_df = pd.concat([n_miss, np.round(ratio, 2), type], axis=1, keys=['n_miss', 'ratio', 'type'])
    missing_df = missing_df[missing_df['ratio']<=drop_thr]
    print(missing_df, end="\n")
    print(len(missing_df))

    if na_name:
        return missing_df.index.tolist()

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def check_num_fillna_results(dataframe):
    temp_df = dataframe.copy()
    na_cols = missing_values_table(temp_df, 100.0, True)
    for col in na_cols:
        if temp_df[col].dtypes in ["float64", "int64"]:
            temp_df[col] = temp_df[col].fillna(temp_df[col].mean())
            plt.subplot(1, 2, 1)
            dataframe[col].hist(bins=20)
            plt.title(col)
            plt.subplot(1, 2, 2)
            temp_df[col].hist(bins=20)
            plt.title(col+'_filled')
            plt.show()
        else:
            temp_df[col] = temp_df[col].fillna(temp_df[col].mode()[0])
            plt.subplot(1, 2, 1)
            sns.countplot(x=dataframe[col], data=dataframe)
            plt.title(col)
            plt.subplot(1, 2, 2)
            sns.countplot(x=temp_df[col], data=temp_df)
            plt.title(col + '_filled')
            plt.show()

def na_operation(dataframe, target, type=0):
    full_na_cols = missing_values_table(dataframe, 100.0, True)
    cols = missing_values_table(dataframe, 10, True)
    drop_cols = [col for col in full_na_cols if (col not in cols) and (col != target)]
    dataframe = dataframe.drop(drop_cols, axis=1)
    if target in cols:
        cols.remove(target)
    if type == 0:
        dataframe.loc[:, cols] = dataframe.loc[:, cols].apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
        dataframe.loc[:, cols] = dataframe.loc[:, cols].apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == "O" else x, axis=0)
    else:
        pass
    return dataframe

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

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

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

def plot_importance(model, features, number_of_tops, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:number_of_tops])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def main():
    ### Task 1: Exploratory Data Analysis
    ## Step 1: Read and combine the Train and Test datasets. Proceed using the merged dataset.
    df_train = pd.read_csv('./train.csv')
    df_test = pd.read_csv('./test.csv')
    df_ = pd.concat([df_train, df_test]).reset_index()
    df = df_.copy()

    ## Step 2: Identify the numerical and categorical variables.
    cat_cols = df.select_dtypes(exclude="number").columns
    num_cols = df.select_dtypes(include="number").columns

    ## Step 3: Make the necessary adjustments. (Such as variables with data type errors.)
    ('DF Head: \n', df.head())
    print('DF Info: \n', df.info())
    res = df_column_check(df)
    print(res)
    for col in cat_cols:
        df[col] = df[col].astype("object")

    ## Step 4: Examine the distribution of numerical and categorical variables in the dataset.
    for col in cat_cols:
        cat_summary(df, col, plot=True)
    for col in num_cols:
        num_summary(df, col, plot=True)

    ## Step 5: Examine the relationship between the categorical variables and the target variable.
    for col in cat_cols:
        target_summary_with_cat(df, "SalePrice", col)

    ## Step 6: Examine whether there are any outliers.
    for col in num_cols:
        print(col, check_outlier(df, col))

    ## Step 7: Examine whether there are any missing values.
    na_cols = missing_values_table(df, 100.0, True)

    ### Görev 2: Feature Engineering
    ## Step 1: Perform the necessary operations for missing values and outliers.
    df = na_operation(df, "SalePrice", type=0)
    cat_cols = df.select_dtypes(exclude="number").columns
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if check_outlier(df, col):
            print(col, check_outlier(df, col))
            df = replace_with_thresholds(df, col)

    ## Step 2: Apply Rare Encoding.
    rare_analyser(df, "SalePrice", cat_cols)
    df = rare_encoder(df, 0.01)

    ## Step 3: Create new variables.
    df['TotalArea'] = df['GarageArea'] + df['PoolArea'] + df['MasVnrArea']
    df['TotalArea_mul'] = df['GarageArea'] * df['PoolArea'] * df['MasVnrArea']

    ## Step 4: Perform the encoding operations.
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]
    max_nunique = max(df[cat_cols].nunique())
    for col in binary_cols:
        df = label_encoder(df, col)
    ohe_cols = [col for col in df.columns if max_nunique >= df[col].nunique() > 2]
    df = one_hot_encoder(df, ohe_cols)

    ### Task 3: Model Building
    ## Step 1: Split the Train and Test datasets. (Observations with missing values in the SalePrice variable belong to the test dataset.)
    X = df.drop(columns=['SalePrice'])
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                    test_size=len(df[df['SalePrice'].isnull()==True]), random_state=17, shuffle=False)

    ## Step 2: Build a model using the training data and evaluate its performance.
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print("Random Forest")
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X=X_train, y=y_train)
    y_pred = rf_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    print('RMSE: ', rmse)
    print('MAE: ', mae)

    # Bonus: Build a model by applying a log transformation to the target variable and observe the RMSE results. Note: Do not forget to take the inverse of the log transformation.
    X = df.drop(columns=['SalePrice'])
    y = np.log1p(df["SalePrice"])
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=len(df[df['SalePrice'].isnull()==True]), random_state=17, shuffle=False)
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print("Random Forest")
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X=X_train, y=y_train)
    y_pred = rf_model.predict(X_train)
    y_train = np.expm1(y_train)
    y_pred = np.expm1(y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    print('RMSE: ', rmse)
    print('MAE: ', mae)

    # Step 3: Perform hyperparameter optimization.
    print(rf_model.get_params())
    rf_params = {"max_depth": [5, 8, 15, 20, 30, None],
                 "max_features": [3, 5, 7, "auto"],
                 "min_samples_split": [2, 5, 8, 15, 20],
                 "n_estimators": [10, 50, 100, 200, 500]}
    rf_gs_best = GridSearchCV(rf_model,
                               rf_params,
                               cv=2,
                               n_jobs=-1,
                               verbose=1).fit(X_train, y_train)
    print(rf_gs_best.best_params_)
    rf_final = rf_model.set_params(**rf_gs_best.best_params_).fit(X_train, y_train)
    y_pred = rf_final.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    print('RMSE: ', rmse)
    print('MAE: ', mae)

    # Step 4: Examine the feature importance levels.
    plot_importance(rf_final, X, 30, False)



if __name__ == '__main__':
    main()