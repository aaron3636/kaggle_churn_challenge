"""
Idea:

Feature Engineering:

Each function takes the df as an input and returns the customer_id with the
newly computed feature
"""

import pandas as pd


def feature_total_pages_count(dataframe: pd.DataFrame):

    dataframe = dataframe.groupby(["userId", "page"]).size()\
        .reset_index(name="count")

    dataframe = dataframe.pivot(index="userId", columns="page", values="count")
    dataframe.fillna(0, inplace=True)

    return dataframe
