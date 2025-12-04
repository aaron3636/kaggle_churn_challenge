import pandas as pd


def feature_total_pages_count(dataframe: pd.DataFrame):
    """
    Docstring for feature_total_pages_count

    :param dataframe: Description
    :type dataframe: pd.DataFrame

    This function takes the df as an input and computes the number of times
    each website was visited in total over the whole observed timeframe
    """

    dataframe = dataframe.groupby(["userId", "page"]).size()\
        .reset_index(name="count")

    dataframe = dataframe.pivot(index="userId", columns="page", values="count")
    dataframe.fillna(0, inplace=True)

    return dataframe


def feature_total_pages_count_1(dataframe: pd.DataFrame):
    """
    Docstring for feature_total_pages_count

    :param dataframe: Description
    :type dataframe: pd.DataFrame

    This function takes the df as an input and computes the number of times
    each website was visited in total over the whole observed timeframe
    """

    dataframe = dataframe.groupby(["userId", "page", "week_slice"]).size()\
        .reset_index(name="count")

    dataframe = dataframe.pivot(
        index=["userId", "week_slice"], columns="page", values="count").reset_index()
    dataframe.fillna(0, inplace=True)

    return dataframe


# TODO
# Track visits over time. For each week. % and Total change
