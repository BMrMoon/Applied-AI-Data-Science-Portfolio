# Imports
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


### Functions
def get_dataset():
    df = pd.read_csv("./amazon_review.csv")
    print('Count Number: \n', df.count())
    print('First Rows: \n', df.head())
    print('Descriptive Statistics: \n', df.describe())
    print('Null Check: \n', df.isnull().sum())
    print('Dataset Insight: \n', df.info())
    return df

def time_based_weighted_average(dataframe, w1=0.24, w2=0.22, w3=0.20, w4=0.18, w5=0.16):
    return (w1*dataframe.loc[dataframe["day_diff"]<30, "overall"].mean()) + \
    (w2*dataframe.loc[(dataframe["day_diff"]<90) & (dataframe["day_diff"]>=30), "overall"].mean()) + \
    (w3*dataframe.loc[(dataframe["day_diff"]<180) & (dataframe["day_diff"]>=90), "overall"].mean()) + \
    (w4*dataframe.loc[(dataframe["day_diff"]<360) & (dataframe["day_diff"]>=180), "overall"].mean()) + \
    (w5*dataframe.loc[dataframe["day_diff"]>=360, "overall"].mean())

def score_pos_neg_diff(positive, negative):
    return positive - negative


def score_average_rating(positive, negative):
    if positive + negative == 0:
        return 0

    return positive / (positive + negative)


def wilson_lower_bound(positive, negative, confidence=0.95):
    """
        Calculate the Wilson Lower Bound Score
	        •	The lower bound of the confidence interval calculated for the Bernoulli parameter p is accepted as the WLB score.
	        •	The calculated score is used for product ranking.
	        •	Note:
        If the scores are on a 1–5 scale, ratings of 1–3 can be marked as negative and 4–5 as positive to make them compatible with the Bernoulli distribution.
        However, this also introduces certain problems. Therefore, it is necessary to use Bayesian average rating instead.

        Parameters
        ----------
        positive: int
            up count
        negative: int
            down count
        confidence: float
            confidence

        Returns
        -------
        wilson score: float

        """
    n = positive + negative
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * positive / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def Rating_Products():
    df = get_dataset()

    ### Task 1: Calculate the average rating based on recent reviews and compare it with the existing average rating.
    ## Step 1: Calculate the average rating of the product.
    print('Average rating for each product: \n', df.groupby('asin').agg({'overall': ['mean']}))

    ## Step 2: Calculate the time-based weighted average rating.
    weight = time_based_weighted_average(df)
    print('Time-based weighted average: \n', weight)

    ## Step 3: Compare and interpret the average of each time period in the weighted rating.
    print('Average rating for smaller than last 30 days: \n', 0.24 * df.loc[df["day_diff"] < 30, "overall"].mean())
    print('Average rating between last 30 to 90 days: \n', 0.22 * df.loc[(df["day_diff"] < 90) & (df["day_diff"] >= 30), "day_diff"].mean())
    print('Average rating between last 90 to 180 days: \n', 0.20 * df.loc[(df["day_diff"] < 180) & (df["day_diff"] >= 90), "overall"].mean())
    print('Average rating between last 180 to 360 days: \n', 0.18 * df.loc[(df["day_diff"] < 360) & (df["day_diff"] >= 180), "overall"].mean())
    print('Average rating for greater than last 360 days: \n', 0.16 * df.loc[df["day_diff"] >= 360, "overall"].mean())

    ### Task 2: Identify the 20 reviews to be displayed on the product detail page.
    ## Step 1: Create the helpful_no variable.
    df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

    ## Step 2: Calculate the score_pos_neg_diff, score_average_rating, and wilson_lower_bound scores, and add them to the dataset.
    df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
    df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
    df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

    ## Step 3: Identify the top 20 reviews and interpret the results.
    print('score_pos_neg_diff top 20 scores: ', df.sort_values("score_pos_neg_diff", ascending=False).head(20))
    print('score_average_rating top 20 scores: ', df.sort_values("score_average_rating", ascending=False).head(20))
    print('wilson_lower_bound top 20 scores:', df.sort_values("wilson_lower_bound", ascending=False).head(20))



def main():
    Rating_Products()

if __name__ == '__main__':
    main()