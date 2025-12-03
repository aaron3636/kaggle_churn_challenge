import pandas as pd


def gender_feature_ont_hot(df: pd.DataFrame):

    df_a = df[["userId", "gender"]].drop_duplicates()

    df_a["gender"] = df_a["gender"] == "M"

    df_a["gender"] = pd.to_numeric(df_a["gender"])

    return df_a
