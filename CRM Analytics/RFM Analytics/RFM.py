### imports
import  pandas as pd

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 500)


### Functions
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, thr:tuple):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, thr[0], thr[1])
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)][col_name].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, thr:tuple):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit.astype(dataframe[variable].dtype)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit.astype(dataframe[variable].dtype)
    return dataframe

def data_preprocessing(dataframe, thr:tuple):
    dataframe["total_shoppping_number"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_amount"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]
    date_cols = dataframe.loc[:,dataframe.columns.str.contains('date')].columns
    for col in date_cols:   
        dataframe[col] = pd.to_datetime(dataframe[col])
    num_cols = dataframe.select_dtypes(include='number').columns.to_list()
    cat_cols = dataframe.select_dtypes(exclude=['number', 'datetime']).columns.to_list()
    
    for col in num_cols:
        if check_outlier(dataframe, col, thr):
            dataframe = replace_with_thresholds(dataframe, col, thr)

    return dataframe

def RFM_Analytics(save:bool):
    ### TASK 1: Data Understanding and Preparation
    ## Step 1: Read the flo_data_20K.csv dataset.
    df_ = pd.read_csv("CRM_Analytics/flo_data_20k.csv")
    df = df_.copy()

    ## Step 2:
    # Step 2.1: The first 10 observations
    print('Dataset Head: \n', df.head(10))

    # Step 2.2: Variable names
    print('Dataset Columns: \n', df.columns)

    # Step 2.3: Descriptive statistics
    print('Descriptive Statistics: \n', df.describe().T)

    # Step 2.4: Missing values
    print('Missing Values: \n', df.isnull().sum())

    # Step 2.5: Examine the variable types
    print('Variable Types: \n')
    [print(f"{col}: ",df[col].dtypes) for col in df.columns]

    ## Step 3: Omnichannel customers are defined as customers who shop through both online and offline platforms. 
    # Create new variables representing each customer’s total number of purchases and total spending.
    df["total_shoppping_number"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_amount"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

    ## Step 4: Examine the variable types. Convert the variables representing dates to the date data type.
    date_cols = df.loc[:,df.columns.str.contains('date')].columns
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    ## Step 5: Examine the distribution of customer counts, average number of items purchased, and average spending 
    # across shopping channels.
    d = {"master_id": "count", "total_shoppping_number": "mean", "total_amount": "mean"}
    print('Average Spending: \n', df.groupby("order_channel").agg(d))

    ## Step 6: Rank the top 10 customers who generate the highest revenue.
    print('Top 10 customers who generate the highest revenuetop 10 customers who generate the highest revenue: \n', df[["master_id", "total_amount"]].sort_values(by=["total_amount"], ascending=False).head(10))

    ## Step 7: Rank the top 10 customers who placed the highest number of orders.
    print('Top 10 customers who placed the highest number of orders: \n', df[["master_id", "total_shoppping_number"]].sort_values(by=["total_shoppping_number"], ascending=False).head(10))

    ## Step 8: Functionalize the data preprocessing step.
    df = data_preprocessing(dataframe=df, thr=(0.01, 0.99))

    ### TASK 2: Calculation of RFM Metrics
    today = max(df['last_order_date']) + pd.Timedelta(days=1)
    rfm = df.groupby("master_id").agg({
                                    "last_order_date": lambda x: (today - x.max()).days,
                                    "total_shoppping_number": lambda x: x,
                                    "total_amount": lambda x: x,
                                    })
    rfm.columns = ['recency', 'frequency', 'monetary']

    ### TASK 3: Calculation of RF and RFM Scores
    rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])
    rfm["rf_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    ### TASK 4: Defining RF Scores as Segments
    seg_map = {
                r'[1-2][1-2]': 'hibernating',
                r'[1-2][3-4]': 'at_Risk',
                r'[1-2]5': 'cant_loose',
                r'3[1-2]': 'about_to_sleep',
                r'33': 'need_attention',
                r'[3-4][4-5]': 'loyal_customers',
                r'41': 'promising',
                r'51': 'new_customers',
                r'[4-5][2-3]': 'potential_loyalists',
                r'5[4-5]': 'champions'
            }
    rfm["segment"] = rfm["rf_score"].replace(seg_map, regex=True)

    ### TASK 5: Time for action!
    ## Step 1: Examination of the segments’ average recency, frequency, and monetary values.
    print('Examination of RFM scores: \n', rfm.groupby("segment").agg({"recency": "mean", "frequency": "mean", "monetary": "mean"}).reset_index())

    ## Step 2: Using RFM analysis, identify customers matching the relevant profiles for the following two cases 
    #and save their customer IDs as CSV files.
    # Step 2.1: FLO is adding a new women’s shoe brand to its portfolio. The prices of this brand are above the 
    # general preferences of its customer base. Therefore, for the promotion of the brand and to support product 
    # sales, FLO wants to personally reach out to customers who fit the relevant profile. The target customers for 
    # this communication are loyal customers (champions, loyal_customers) who have an average spending above 250 TL 
    # and have shopped in the women’s category. Save the customer IDs of these customers as a CSV file named 
    # new_brand_target_customer_ids.csv.
    df_a = df[df["interested_in_categories_12"].str.contains("KADIN")]
    df_a[df_a["total_amount"]/df_a["total_shoppping_number"] > 250]["master_id"].reset_index(drop=True)
    print('Champion or loyal customers: \n', df_a)
    if save:
        df_a.to_csv("yeni_marka_hedef_müşteri_id.csv")

    # Step 2.2: A discount of nearly 40% is planned for men’s and children’s products. For this campaign, FLO 
    # wants to specifically target customers who are interested in these categories and who were previously valuable 
    # customers but have not shopped for a long time, customers who are at risk of being lost, dormant customers, and 
    # new customers. Save the IDs of customers matching this profile as a CSV file named 
    # discount_target_customer_ids.csv.
    df_b = df[(df["interested_in_categories_12"].str.contains("ERKEK"))
                & (df["interested_in_categories_12"].str.contains("COCUK"))]
    target = rfm[(rfm["segment"]=="cant_loose") | (rfm["segment"]=="hibernating") | (rfm["segment"]=="new_customers")]
    df_b["rfm_condition"] = df_b.groupby(df_b.index).agg({"master_id": lambda x: True if x.item() in [idx for idx in target.index] else False})
    df_b[df_b["rfm_condition"]==True].reset_index()["master_id"]
    print('Customers in risk: \n', df_b)
    if save:
        df_b.to_csv("indirim_hedef_müşteri_ids.csv")



def main():
    RFM_Analytics(False)

if __name__ == '__main__':
    main()