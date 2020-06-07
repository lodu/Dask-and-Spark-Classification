import pandas as pd
from pymongo import MongoClient

from config import password, port, user


def _csv_to_df(file_name):
    df = pd.read_csv(file_name, sep=",", na_values=["No Positive", "No Negative"])
    df.head()
    return df


# Remove all columns but Positive_Review and Negative_Review
def _remove_all_but_review_texts(df):
    columns = list(df.columns)
    if "Positive_Review" in columns:
        columns.remove("Positive_Review")
    if "Negative_Review" in columns:
        columns.remove("Negative_Review")
    return df.drop(columns=columns)


def _concat_all_reviews_with_label_column(df):
    negative_reviews = df.loc[:, ["Negative_Review"]]
    positive_reviews = df.loc[:, ["Positive_Review"]]

    positive_reviews["label"] = 1
    negative_reviews["label"] = 0

    renamed_positive_reviews = positive_reviews.rename(
        {"Positive_Review": "review"}, axis=1
    )
    renamed_negative_reviews = negative_reviews.rename(
        {"Negative_Review": "review"}, axis=1
    )

    return renamed_positive_reviews, renamed_negative_reviews


def _uncleaned_df_to_cleaned_df(uncleaned_df):
    df = _remove_all_but_review_texts(uncleaned_df)
    positive, negative = _concat_all_reviews_with_label_column(df)

    # Remove all 'No Negative" and 'No Positive" reviews-texts
    negative = negative.loc[lambda x: x["review"] != "No Negative"]
    positive = positive.loc[lambda x: x["review"] != "No Positive"]
    return positive, negative


def uncleaned_csv_to_cleaned_df(file_name):
    df = _csv_to_df(file_name)
    positive, negative = _uncleaned_df_to_cleaned_df(df)
    return df, positive, negative


def add_label_to_uncleaned_df(df):
    columns_to_keep = [
        "Hotel_Address",
        "Hotel_Name",
        "lat",
        "lng",
        "Average_Score",
        "Total_Number_of_Reviews",
        "Additional_Number_of_Scoring",
        "Reviewer_Nationality",
        "Review_Date",
        "Total_Number_of_Reviews_Reviewer_Has_Given",
        "Reviewer_Score",
        "Tags",
    ]
    temp_columns_to_keep = columns_to_keep
    temp_columns_to_keep.extend(
        ["Negative_Review", "Review_Total_Negative_Word_Counts"]
    )

    negative_reviews = df.loc[:, temp_columns_to_keep]
    temp_columns_to_keep = [
        x
        for x in temp_columns_to_keep
        if x not in ["Negative_Review", "Review_Total_Negative_Word_Counts"]
    ]

    temp_columns_to_keep.extend(
        ["Positive_Review", "Review_Total_Positive_Word_Counts"]
    )
    positive_reviews = df.loc[:, temp_columns_to_keep]
    temp_columns_to_keep = [
        x
        for x in temp_columns_to_keep
        if x not in ["Positive_Review", "Review_Total_Positive_Word_Counts"]
    ]

    positive_reviews["Sentiment"] = 1
    negative_reviews["Sentiment"] = 0

    renamed_positive_reviews = positive_reviews.rename(
        {
            "Positive_Review": "Review",
            "lat": "Lat",
            "lng": "Lng",
            "Review_Total_Positive_Word_Counts": "Review_Word_Counts",
        },
        axis=1,
    )
    renamed_negative_reviews = negative_reviews.rename(
        {
            "Negative_Review": "Review",
            "lat": "Lat",
            "lng": "Lng",
            "Review_Total_Negative_Word_Counts": "Review_Word_Counts",
        },
        axis=1,
    )

    long_df = pd.concat([renamed_positive_reviews, renamed_negative_reviews], axis=0)
    return long_df


# Creates (1) all csv reviews into dataframe (2) only positive reviews (3) only negative reviews
uncleaned_df, positive_df, negative_df = uncleaned_csv_to_cleaned_df(
    r"Hotel_Reviews.csv"
)
uncleaned_df.head()

# Write negative and positive dataframes to CSV (review, label)
positive_df.to_csv(r"Review_pos.csv", index=False)
negative_df.to_csv(r"Review_neg.csv", index=False)


long_df = add_label_to_uncleaned_df(uncleaned_df)


mongo_client = MongoClient(
    f"mongodb://{user}:{password}@localhost:{port}/?authSource=admin&readPreference=primary"
)

# Collection with a lot of dataset columns
db = mongo_client.reviews
if db.long.count() < 100:
    db.long.insert_many(long_df.to_dict("records"))
else:
    print(f"Didn't insert into long because already size: {int(db.long.count())}")

# Collection with only review and sentiment (label)
short_df = long_df.loc[:, ["Review", "Sentiment"]]
if db.short.count() < 100:
    db.short.insert_many(short_df.to_dict("records"))
else:
    print(f"Didn't insert into short because already size: {int(db.long.count())}")
