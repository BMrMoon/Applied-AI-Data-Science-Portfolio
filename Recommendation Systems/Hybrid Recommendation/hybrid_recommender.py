##### Hybrid Recommender System #####



### Business Problem
## Recommend 10 movies for the user with the given ID by using item-based and user-based recommendation methods.

### Story of the Dataset
## The dataset was provided by MovieLens, a movie recommendation service. It contains movies along with the rating scores given to them. It includes 20,000,263 ratings across 27,278 movies. This particular dataset was created on October 17, 2016 and contains data from 138,493 users between January 9, 1995 and March 31, 2015. The users were selected randomly, and it is known that each selected user has rated at least 20 movies.

## movie.csv
# movieId: Unique movie ID.
# title: Movie title
# genres: Genre

## rating.csv
# userid: Unique user ID.
# movieId: Unique movie ID.
# rating: Rating given to the movie by the user
# timestamp: Rating date



# Imports
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)
#pd.set_option("display.width", 1500)
#pd.set_option("display.expand_frame_repr", False)


# Functions
def get_pivot_table():
    movie = pd.read_csv('./Hybrid_Recommender_System/datasets/movie.csv')
    rating = pd.read_csv("./Hybrid_Recommender_System/datasets/rating.csv")
    rating = rating.merge(movie, how="left", on="movieId")
    movies_count = pd.DataFrame(rating["title"].value_counts()).reset_index()
    low_rated_movies = movies_count.loc[movies_count["count"] < 1000].drop("count", axis=1)
    rating = rating[~rating["title"].isin(low_rated_movies["title"])]
    return  rating.pivot_table(index=["userId"], columns=["title"], values="rating")

def user_based_and_item_based_recommender():
    #### User Based Recommendation
    ### Task 1: Data Preparation
    ## Step 1: Read the movie and rating datasets.
    movie = pd.read_csv('./Hybrid_Recommender_System/datasets/movie.csv')
    rating = pd.read_csv("./Hybrid_Recommender_System/datasets/rating.csv")

    ## Step 2: Add the movie titles and genres corresponding to the IDs in the rating dataset from the movie dataset.
    rating = rating.merge(movie, how="left", on="movieId")

    ## Adım 3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.
    movies_count = pd.DataFrame(rating['title'].value_counts()).reset_index()
    low_rated_movies = movies_count.loc[movies_count["count"] < 1000].drop("count", axis=1)
    rating =rating[~rating["title"].isin(low_rated_movies["title"])]

    ## Step 4: Create a pivot table for a dataframe where userIDs are in the index, movie titles are in the columns, and ratings are the values.
    p_table = get_pivot_table()

    ## Step 5: Functionalize all the steps performed.

    ### Task 2: Identifying the Movies Watched by the User to Be Recommended For
    ## Step 1: Select a random user ID.
    user_movie_df = get_pivot_table()
    random_user = np.int64(pd.Series(rating["userId"].values).sample(n=45, random_state=1).values)

    ## Step 2: Create a new dataframe named random_user_df consisting of the observation units belonging to the selected user.
    random_user_df = user_movie_df[user_movie_df.index==random_user[0]]

    ## Step 3: Assign the movies rated by the selected user to a list named movies_watched.
    movies_watched = random_user_df.columns[random_user_df.notna().any()]

    ### Task 3: Accessing the Data and IDs of Other Users Who Watched the Same Movies
    ## Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturunuz.
    movies_watched_df = user_movie_df[movies_watched]

    ## Step 2: Create a new dataframe named user_movie_count that contains, for each user, the number of movies watched by the selected user that they have also watched.
    user_movie_count = pd.DataFrame(movies_watched_df.T.notnull().sum(), columns=["count"])

    ## Step 3: Create a list named users_same_movies containing the user IDs of those who have watched 60% or more of the movies rated by the selected user.
    users_same_movies = user_movie_count.reset_index(drop=False)
    users_same_movies['count_perc'] = users_same_movies['count']/len(movies_watched)*100
    users_same_movies = users_same_movies[(users_same_movies["count_perc"]>60)].drop(columns=['count'])

    ### Task 4: Identifying the Users Most Similar to the User to Be Recommended For
    ## Step 1: Filter the movies_watched_df dataframe so that it contains the IDs of users in the users_same_movies list who are similar to the selected user.
    final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies["userId"])]

    ## Step 2: Create a new dataframe named corr_df that will contain the correlations between users.
    corr_df = pd.DataFrame(final_df.T.corr().unstack().sort_values().drop_duplicates(), columns=["corr"])
    corr_df.index.names = ["userId_1", "userId_2"]
    corr_df = corr_df.reset_index()

    ## Step 3: Filter the users who have a high correlation with the selected user (greater than 0.65) and create a new dataframe named top_users.
    top_users = corr_df[(corr_df["userId_1"] == random_user[0]) & (corr_df["corr"] >= np.quantile(corr_df["corr"], .50))][["userId_2", "corr"]].reset_index(drop=True)
    top_users.columns = ["userId", "corr"]

    ## Step 4: Merge the top_users dataframe with the rating dataset.
    top_users = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

    ### Task 5: Calculating the Weighted Average Recommendation Score and Retaining the Top 5 Movies
    ## Step 1: Create a new variable named weighted_rating, which is obtained by multiplying each user’s corr and rating values.
    top_users["weighted_rating"] = top_users["rating"] * top_users["corr"]

    ## Step 2: Create a new dataframe named recommendation_df that contains the movie IDs and the average weighted ratings of all users for each movie.
    recommendation_df = top_users.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()

    ## Step 3: Select the movies in recommendation_df with a weighted_rating greater than 3.5 and sort them by weighted_rating.
    movie_ids_list = recommendation_df.sort_values(by="weighted_rating", ascending=False)["movieId"][0:5].values.tolist()

    ## Step 4: Retrieve the movie titles from the movie dataset and select the top 5 movies to recommend.

    print('Recommended Top 5 Movies by User Based Recommender: \n', movie[movie["movieId"].isin(movie_ids_list)]["title"], '\n')

    #### Item Based Recommendation
    ### Task 1: Make an item-based recommendation based on the most recent movie watched and the highest-rated movie by the user.
    ## Step 1: Read the movie and rating datasets.
    movie = pd.read_csv('./Hybrid_Recommender_System/datasets/movie.csv')
    rating = pd.read_csv("./Hybrid_Recommender_System/datasets/rating.csv")

    ## Step 2: Retrieve the ID of the most recently rated movie among the movies that the selected user rated 5 points.
    movie_id = rating[(rating["userId"] == random_user[0]) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

    ## Step 3: Filter the user_movie_df dataframe created in the user-based recommendation section according to the selected movie ID.
    last_movie = movie[movie["movieId"] == movie_id]["title"].values[0]
    filtered_user_movie_df = user_movie_df[last_movie]

    ## Step 4: Using the filtered dataframe, calculate and rank the correlations between the selected movie and the other movies.
    corr_df = user_movie_df.corrwith(filtered_user_movie_df).sort_values(ascending=False)[1:-1].reset_index(drop=False)
    corr_df.columns = ['title', 'corr']
    corr_df = corr_df.drop(columns=['corr'])

    ## Step 5: Recommend the top 5 movies excluding the selected movie itself.
    print('Recommended Top 5 Movies by Item Based Recommender: \n', corr_df.head(5))



def main():
    user_based_and_item_based_recommender()

if __name__ == '__main__':
    main()
