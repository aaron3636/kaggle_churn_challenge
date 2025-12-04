from skrub import GapEncoder
import pandas as pd


def gap_encoder_location(df: pd.DataFrame, gec: GapEncoder):

    gec.fit(pd.Series(df["location"].unique(), name="location"))

    encoded_df = gec.transform(df["location"])

    df_a = pd.concat((df, encoded_df), axis=1)

    X_columns = []

    for elem in df_a.columns:
        if "location:" in elem:
            X_columns.append(elem)

    df_b = df_a[X_columns]

    return df_b


def location_state_one_hot_encoded(df: pd.DataFrame):

    df["state"] = df["location"].apply(lambda s: s.split(",")[-1].strip())

    s = df["state"].str.get_dummies(sep="-")  # columns like CA, NY, NJ, ...

    df = df[["userId"]].join(s)

    df = df.drop_duplicates()

    return df


def location_state_one_hot_encoded_1(df: pd.DataFrame):

    df["state"] = df["location"].apply(lambda s: s.split(",")[-1].strip())

    s = df["state"].str.get_dummies(sep="-")  # columns like CA, NY, NJ, ...

    out = pd.concat([df[["userId", "week_slice"]], s], axis=1)

    # one row per (userId, week_slice); max = "did this state appear at least once"
    out = out.groupby(["userId", "week_slice"], as_index=False).max()

    return out
