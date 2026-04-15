# Imports
import  pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

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

def CLTV_Analytics():
    ### TASK 1: Data Preparation
    ## Step 1: Read the flo_data_20K.csv dataset.
    df_ = pd.read_csv("/Users/berkaybey/Code/AI Data Scientist Bootcamp/pythonProject/Case_Studies/CRM_Analytics/CLTV_Analytics/flo_data_20k.csv")
    df = df_.copy()

    ## Step 2: Define the outlier_thresholds and replace_with_thresholds functions required to suppress outliers.

    ## Step 3: If there are any outliers in the variables "order_num_total_ever_online", 
    #"order_num_total_ever_offline", "customer_value_total_ever_offline", and  "customer_value_total_ever_online", 
    #suppress them.
    df = data_preprocessing(dataframe=df, thr=(0.01, 0.99))

    ## Step 4: Omnichannel customers are defined as customers who shop through both online and offline platforms. Create 
    #new variables for each customer’s total number of purchases and total spending.
    df["total_shoppping_number"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_amount"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    
    ## Step 5: Examine the variable types. Convert the variables representing dates to the date data type.
    print('General Insight: \n', df.info())
    date_cols = df.loc[:,df.columns.str.contains('date')].columns
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    ### Task 2: Creation of the CLTV Data Structure
    ## Step 1: Use the date that is 2 days after the most recent purchase in the dataset as the analysis date.
    today = max(df['last_order_date']) + pd.Timedelta(days=2)

    ## Step 2: Create a new CLTV dataframe that includes the variables customer_id, recency_cltv_weekly, T_weekly, 
    #frequency, and monetary_cltv_avg. The monetary value should be expressed as the average value per purchase, while 
    #recency and tenure values should be expressed in weeks.
    recency_df = pd.DataFrame()
    recency_df["recency"] = (df["last_order_date"] - df["first_order_date"]).apply(lambda x: x.days)
    recency_df.index = df["master_id"]
    cltv_df = df.groupby('master_id').agg({"first_order_date": lambda x: (today - min(x)).days,
                                        'total_shoppping_number': lambda x: x,
                                        'total_amount': lambda x: x.sum()})
    cltv_df = pd.merge(recency_df, cltv_df, left_index=True, right_index=True)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["frequency"] = cltv_df["frequency"].astype(int)
    cltv_df["monetary"] = cltv_df["monetary"]/cltv_df["frequency"]
    cltv_df["recency"] = (cltv_df["recency"]/7)
    cltv_df["T"] = cltv_df["T"]/7
    cltv_df = cltv_df[cltv_df["frequency"]>1]
    cltv_df.reset_index(inplace=True)

    ### Task 3: Building the BG/NBD and Gamma-Gamma Models and Calculating CLTV
    ## Step 1: Fit the BG/NBD model.
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

    # Step 1.1: Predict the expected purchases from customers over the next 3 months and add them to the CLTV dataframe 
    #as exp_sales_3_month.
    cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(t=12, frequency=cltv_df["frequency"], recency=cltv_df["recency"], T=cltv_df["T"])

    # Step 1.2: Predict the expected purchases from customers over the next 6 months and add them to the CLTV dataframe 
    #as exp_sales_6_month.
    cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(t=24, frequency=cltv_df["frequency"], recency=cltv_df["recency"], T=cltv_df["T"])

    ## Step 2: Fit the Gamma-Gamma model. Predict customers’ 
    #expected average value and add it to the CLTV dataframe as exp_average_value.
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary'])

    ## Step 3: Calculate the 6-month CLTV and add it to the dataframe as cltv.
    # Step 3.1: Observe the top 20 customers with the highest CLTV values.
    cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
    cltv_df.head(20)

    ### Task 4: Creating Segments Based on CLTV Value
    ## Step 1: Segment all customers into 4 groups based on 6-month CLTV and add the segment names to the dataset.
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    ## Step 2: For 2 of the 4 groups you select, provide brief 6-month action recommendations for management.
    print(cltv_df.groupby("cltv_segment").agg({"cltv": ["mean", "std", "min", "max"], "master_id": "nunique"}))
    print('Top 20: \n', cltv_df.sort_values(["cltv"], ascending=False).head(20))



def main():
    CLTV_Analytics()

if __name__ == '__main__':
    main()