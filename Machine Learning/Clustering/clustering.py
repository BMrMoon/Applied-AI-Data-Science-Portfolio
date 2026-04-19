# Imports
from cmath import inf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


# Functions
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)][col_name].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit.astype(dataframe[variable].dtype)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit.astype(dataframe[variable].dtype)
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
def main():
    ### Task 1: Data Preparation
    ## Step 1: Read the flo_data_20K.csv dataset.
    df_ = pd.read_csv('./flo_data_20k 2.csv')
    df = df_.copy()

    ## Step 2: Select the variables you will use to segment the customers. Note: You can create new variables such as Tenure (customer age) and Recency (how many days ago the customer made their most recent purchase).
    df.info()
    date_cols = df.loc[:,df.columns.str.contains('date')].columns
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    df['day_diff'] = (df['last_order_date'] - df['first_order_date']).dt.days +1
    df['daily_order_average'] = (df['order_num_total_ever_online'] + df['order_num_total_ever_offline']) / df['day_diff']
    df['daily_average_spent'] = (df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']) / df['day_diff']
    today = max(df['last_order_date']) + pd.Timedelta(days=1)
    df['tenure'] = (today - df['first_order_date']).dt.days
    df['recency'] = (today - df['last_order_date']).dt.days

    num_cols = df.select_dtypes(include=['number']).columns.to_list()
    cat_cols = df.select_dtypes(exclude=['number', 'datetime64']).drop(columns=['master_id']).columns.to_list()

    # print(df[num_cols].describe().T)
    # for col in num_cols:
    #     if check_outlier(df, col):
    #         df = replace_with_thresholds(df, col)
    # print(df[num_cols].describe().T)

    ### Task 2: Customer Segmentation with K-Means
    ## Step 1: Standardize the variables.
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    binary_cols = [col for col in df.columns if col in cat_cols
                   and df[col].nunique() == 2]
    for col in binary_cols:
        df = label_encoder(df, col)
    ohe_cols = [col for col in df.columns if (df[col].nunique() > 2) & (col in cat_cols)]
    df = one_hot_encoder(df, ohe_cols)

    ## Step 2: Determine the optimal number of clusters.
    K = list(range(1, 21))

    ## Step 3: Build your model and segment your customers.
    ssd = []
    kmeans_model = KMeans()
    for k in K:
        kmeans_model = KMeans(n_clusters=k, random_state=24).fit(df.drop(columns=date_cols).drop(columns=['master_id']))
        ssd.append(kmeans_model.inertia_)
    plt.plot(K, ssd, "bx-")
    plt.ylabel("SSE/SSR/SSD")
    plt.xlabel("K")
    plt.title("Optimum Küme sayısı için Elbow Yöntemi")
    plt.show()
    kmeans_model = KMeans()
    elbow = KElbowVisualizer(kmeans_model, k=(K[0], K[-1]), force_model=True)
    elbow.fit(df.drop(columns=date_cols).drop(columns=['master_id']))
    elbow.show()
    print('Estimated number of clusters: ', elbow.elbow_value_)
    kmeans_final = KMeans(n_clusters=elbow.elbow_value_, random_state=24).fit(df.drop(columns=date_cols).drop(columns=['master_id']))
    print("Labels: ", kmeans_final.labels_)
    df['clusters'] = kmeans_final.labels_
    num_cols = df.select_dtypes(include=['number']).columns.to_list()

    ## Step 4: Examine each segment statistically.
    d_statistics_k = df[num_cols].groupby('clusters').agg(["count","mean","median"])
    print('------- Results of KMeans Clustering -------')
    for col in num_cols:
        if col != 'clusters':
            print(f"Descriptive Statistics of {col}: \n", df[[col, 'clusters']].groupby('clusters').agg(["count","mean","median"]), '\n\n')

    ### Task 3: Customer Segmentation with Hierarchical Clustering
    ## Step 1: Determine the optimal number of clusters using the standardized dataframe from Task 2.
    hc_average = linkage(df.drop(columns=date_cols).drop(columns=['master_id']), "average")
    plt.figure(figsize=(10, 5))
    plt.xlabel("Observations")
    plt.ylabel("Distances")
    dendrogram(hc_average,
               leaf_font_size=10)
    plt.show()

    ## Step 2: Build your model and segment your customers.
    h_cluster = AgglomerativeClustering(n_clusters=kmeans_final.n_clusters, linkage="average")
    h_clusters = h_cluster.fit_predict(df.drop(columns=date_cols).drop(columns=['master_id']))
    df['h_clusters'] = h_clusters
    num_cols = df.select_dtypes(include=['number']).columns.to_list()

    ## Step 3: Examine each segment statistically.
    d_statistics_h = df[num_cols].groupby('h_clusters').agg(["count","mean","median"])
    print('------- Results of Hierarchical Clustering -------')
    for col in num_cols:
        if col != 'h_clusters':
            print(f"Descriptive Statistics of {col}: \n", df[[col, 'h_clusters']].groupby('h_clusters').agg(["count","mean","median"]), '\n\n')



if __name__ == '__main__':
    main()