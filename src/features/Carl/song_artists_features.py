
import pandas as pd


def num_different_songs(df: pd.DataFrame):

    df_a = df[["userId", "song"]].drop_duplicates()

    df_a = df_a.groupby("userId").size().reset_index(name="num_different_songs")

    return df_a


def num_different_artists(df: pd.DataFrame):

    df_a = df[["userId", "artist"]].drop_duplicates()

    df_a = df_a.groupby(["userId"]).size().reset_index(name="num_different_artists")

    return df_a


def num_different_songs_1(df: pd.DataFrame):

    df_a = df[["userId", "week_slice", "song"]].drop_duplicates()

    df_a = df_a.groupby(["userId", "week_slice"]).size(
    ).reset_index(name="num_different_songs")

    return df_a


def num_different_artists_1(df: pd.DataFrame):

    df_a = df[["userId", "week_slice", "artist"]].drop_duplicates()

    df_a = df_a.groupby(["userId", "week_slice"]).size(
    ).reset_index(name="num_different_artists")

    return df_a
