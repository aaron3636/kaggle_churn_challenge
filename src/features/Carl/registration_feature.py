import datetime
import pandas as pd


def member_since(df: pd.DataFrame):

    df["today"] = df["date"].max()

    df["today"] = pd.to_datetime(df["today"])
    df["registration"] = pd.to_datetime(df["registration"])

    df["member_since"] = df["today"] - df["registration"]

    df["member_since"] = df["member_since"].dt.days

    df = df[["userId", "member_since"]]

    df = df.drop_duplicates()

    return df
