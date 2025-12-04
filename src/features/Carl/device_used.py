import pandas as pd
import numpy as np


def device_type(s: str):
    if "Windows" in s:
        return "Windows"
    elif "Macintosh" in s:
        return "Macintosh"
    elif "Linux" in s:
        return "Linux"
    elif "iPad" in s:
        return "iPad"
    elif "iPhone" in s:
        return "iPhone"
    else:
        return "Different Device"


def exact_device_type(df: pd.DataFrame):

    df["device_used"] = df["userAgent"].apply(
        lambda x:  x[x.index("("): x.index(")") + 1])

    df["exact_device"] = df["device_used"].apply(device_type)

    df_a = df[["userId", "exact_device"]]

    df_b = df_a.drop_duplicates()

    df_b.loc[:, "val"] = np.ones(len(df_b))

    df_c = df_b.pivot(index="userId", columns="exact_device", values="val")

    df_c = df_c.fillna(0.0)

    return df_c


def exact_device_type_1(df: pd.DataFrame):

    df["device_used"] = df["userAgent"].apply(
        lambda x:  x[x.index("("): x.index(")") + 1])

    df["exact_device"] = df["device_used"].apply(device_type)

    df_a = df[["userId", "exact_device", "week_slice"]]

    df_b = df_a.drop_duplicates()

    df_b.loc[:, "val"] = np.ones(len(df_b))

    df_c = df_b.pivot(index=["userId", "week_slice"],
                      columns="exact_device", values="val").reset_index()

    df_c = df_c.fillna(0.0)

    return df_c
