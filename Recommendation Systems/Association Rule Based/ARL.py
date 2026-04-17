# Imports
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


# Functions
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

def ARL():
    ### Task 1: Data Preparation
    ## Step 1: Read the armut_data.csv file.
    df_ = pd.read_csv("./Association_Rule_Based/armut_data.csv")
    df = df_.copy()

    ## Step 2: Since ServiceID represents a different service within each CategoryID, create a new variable to represent these services by combining ServiceID and CategoryID with "_".
    df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
    
    ## Step 3: The dataset consists of the date and time when services were purchased, and there is no basket definition (such as an invoice). In order to apply Association Rule Learning, a basket definition must be created. Here, the basket is defined as the services each customer purchased on a monthly basis. For example, for the customer with ID 7256, the services 9_4 and 46_4 purchased in August 2017 represent one basket, while the services 9_4 and 38_4 purchased in October 2017 represent another basket. These baskets must be identified with a unique ID. For this, first create a new date variable containing only the year and month. Then, combine UserID and the newly created date variable with "_" and assign it to a new variable called ID.
    df["New_Date"] = pd.to_datetime(df["CreateDate"]).apply(lambda x: f"{x.year}-{x.month}")
    df["SepetID"] = df["UserId"].astype(str) + "_" + df["New_Date"].astype(str)

    ### Task 2: Generate Association Rules and Make Recommendations
    ## Step 1: Create a basket-service pivot table as shown below.
    rules_df = df.groupby(['SepetID', 'Hizmet'])["Hizmet"].unique().unstack().fillna(0).apply(lambda x: list(map(lambda y: 1 if y != 0 else 0, x)))
    rules_df = rules_df.astype(bool)

    ## Step 2: Generate the association rules.
    frequent_itemsets = apriori(rules_df,
                            min_support=0.01,
                            use_colnames=True)

    frequent_itemsets.sort_values("support", ascending=False)

    rules = association_rules(frequent_itemsets,
                              metric="support",
                              min_threshold=0.01)
    rules.sort_values("lift", ascending=False)

    ## Step 3: Using the arl_recommender function, make service recommendations for a user who most recently purchased the 2_0 service.
    recomendations = arl_recommender(rules, "2_0", 2)
    print("Recomendations: \n", recomendations)



def main():
    ARL()

if __name__ == '__main__':
    main()