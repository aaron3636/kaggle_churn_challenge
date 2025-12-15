"""Generate user groups by their top-listened artist.

This script computes, for each user, the artist with the most song plays
(`NextSong` events), then aggregates users by that top artist.
"""
from pathlib import Path
import pandas as pd


def build_top_artist_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-user top artist and aggregated groups per artist.

    top_per_user: columns userId, artist, plays (top plays), total_plays, top_share
    group_summary: per-artist aggregation with user count and share stats
    """
    plays = df[df["page"] == "NextSong"][["userId", "artist"]].dropna()
    # Count plays per user/artist
    user_artist_counts = (
        plays.groupby(["userId", "artist"]).size().rename("plays").reset_index()
    )

    # Pick each user's top artist (max plays); idxmax keeps the first in case of tie
    top_idx = user_artist_counts.groupby("userId")["plays"].idxmax()
    top_per_user = user_artist_counts.loc[top_idx].copy()

    # Total plays per user to compute share
    total_plays = plays.groupby("userId").size().rename("total_plays")
    top_per_user = top_per_user.join(total_plays, on="userId")
    top_per_user["top_share"] = top_per_user["plays"] / top_per_user["total_plays"].replace(0, pd.NA)

    # Aggregate by artist: how many users have this artist as their top, plus share stats
    group_summary = (
        top_per_user.groupby("artist")
        .agg(
            users=("userId", "nunique"),
            total_top_plays=("plays", "sum"),
            median_top_share=("top_share", "median"),
            mean_top_share=("top_share", "mean"),
        )
        .sort_values("users", ascending=False)
    )

    return top_per_user.reset_index(drop=True), group_summary.reset_index()


def main():
    root = Path(__file__).resolve().parents[3]
    data_path = root / "data" / "churn-prediction-25-26" / "train.parquet"
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path, columns=["userId", "artist", "page"])

    top_per_user, group_summary = build_top_artist_groups(df)

    print("\nTop artists by number of users (top 15):")
    print(group_summary.head(15).to_string(index=False, formatters={"mean_top_share": "{:.3f}".format, "median_top_share": "{:.3f}".format}))

    # Show sample users for the most common top artist
    if not group_summary.empty:
        top_artist = group_summary.iloc[0]["artist"]
        sample_users = top_per_user[top_per_user["artist"] == top_artist].head(10)
        print(f"\nSample users whose top artist is '{top_artist}':")
        print(sample_users.to_string(index=False))


if __name__ == "__main__":
    main()
