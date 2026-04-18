# Imports
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Functions
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

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

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

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def feature_engineering():
    df_ = pd.read_csv('./diabetes.csv')
    df = df_.copy()

    ### Task 1: Exploratory Data Analysis
    ## Step 1: Examine the overall picture.
    print(df.head(10))
    print(df.describe().T)
    print(df.info())
    print(df.shape)
    print(df.isnull().sum())
    for col in df.columns:
        print(df[df[col] == 0].shape)
    print(df.any(axis=1))

    ## Step 2: Identify the numerical and categorical variables.
    num_cols = [col for col in df.columns if (df[col].dtype in ['float', 'int']) & (df[col].nunique() >= 10)]
    cat_cols = [col for col in df.columns if col not in num_cols]

    ## Step 3: Analyze the numerical and categorical variables.
    for col in num_cols:
        num_summary(df, col, plot=True)
    for col in cat_cols:
        cat_summary(df, col, plot=True)

    ## Step 4: Perform target variable analysis. (Analyze the mean of the target variable by categorical variables, and the mean of numerical variables by the target variable.)
    for col in num_cols:
        target_summary_with_num(df, 'Outcome', col)
    for col in cat_cols:
        target_summary_with_cat(df, 'Outcome', col)

    ## Step 5: Perform outlier analysis.
    print(df.describe().T)
    for col in num_cols:
        print(col, check_outlier(df, col))

    ## Step 6: Perform missing value analysis.
    print(df.isnull().sum())

    ## Step 7: Perform correlation analysis.
    drop_list = high_correlated_cols(df, plot=True)

    ### Task 2: Feature Engineering
    ## Step 1: Perform the necessary operations for missing and outlier values. Although the dataset does not contain missing observations, values of 0 in variables such as Glucose and Insulin may indicate missing values. For example, a person’s glucose or insulin level cannot realistically be 0. Taking this into account, you can first assign 0 values in the relevant variables as NaN and then apply the necessary procedures for missing values.
    possible_nan_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in possible_nan_cols:
        df[col] = df[col].replace(0, np.nan)

    msno.bar(df)
    plt.show()

    msno.matrix(df)
    plt.show()

    msno.heatmap(df)
    plt.show()

    na_cols = missing_values_table(df, True)
    missing_vs_target(df, "Outcome", na_cols)

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    ## Step 2: Create new variables.
    df['G_per_I'] = df['Glucose'] / df['Insulin']
    df.describe().T
    df = df.drop(index=df[df['G_per_I']==np.inf].index)

    df['P_mul_B'] = df['Pregnancies'] * df['BMI']
    df['A_mul_G'] = df['Age'] * df['Glucose']
    df['G_mul_I'] = df['Glucose'] * df['Insulin']


    ## Step 3: Perform the encoding operations.

    ## Step 4: Standardize the numerical variables.
    scaler = RobustScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    ## Step 5: Build the model.
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
    rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_pred, y_test))
    plot_importance(rf_model, X_train, len(X))



def main():
    feature_engineering()

if __name__ == '__main__':
    main()