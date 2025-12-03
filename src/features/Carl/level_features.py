import pandas as pd


def last_level_of_each_customer(df: pd.DataFrame):

    # TODO This is probably wrong
    df["last_time"] = df.groupby(["userId"])["time"].transform("max")

    df["last_item"] = df.groupby(["userId", "sessionId"])[
        "itemInSession"].transform("max")

    df = df[(df["time"] == df["last_time"]) &
            (df["itemInSession"] == df["last_item"])][["userId", "level"]]

    df["level"] = df["level"] == "paid"
    df["level"] = df["level"].astype(int)

    return df
