import pandas as pd
from .utils import weekly_features


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


def feature_page_dynamic(df: pd.DataFrame, page: str, rolling_time_window=1):
    # df_pages = df[df["page"] == "Downgrade"]

    df_pages = df[df["page"] == page]

    df_pages_downgrade = df_pages.groupby(["userId", "date"])[
        "page"].count().reset_index(name=f"number_of_{page}")

    df_pages_downgrade_rolling = weekly_features(
        df=df_pages_downgrade,
        feature=f"number_of_{page}",
        rolling_time_window=rolling_time_window)

    df_res = pd.DataFrame(
        data={
            "userId": df["userId"].unique(),
        })

    df_res = df_res.merge(df_pages_downgrade_rolling, on="userId", how="left")

    df_res = df_res.fillna(0.0)

    return df_res


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
