import pandas as pd


def num_status(df: pd.DataFrame):

    df_a = df.groupby(["userId", "status"]).size().reset_index(name="num")

    df_a["status"] = df_a["status"].apply(str)

    df_b = df_a.pivot(index="userId", columns="status", values="num")

    df_b["404"] = df_b["404"].fillna(0.0)
    df_b["307"] = df_b["307"].fillna(0.0)
    df_b["200"] = df_b["200"].fillna(0.0)

    return df_b
