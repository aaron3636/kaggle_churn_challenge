"""Regenerate the small synthetic fixtures used by the test suite.

sample_events.csv is a hand-built raw event log (same shape as the real Kaggle data,
but tiny and made up). sample_users.csv is derived from it by running the real
feature-engineering pipeline, so the two fixtures always stay consistent.

Run: uv run python tests/fixtures/make_fixtures.py
"""

from pathlib import Path

import pandas as pd

from churn.data import clean_data
from churn.model import build_training_table

FIXTURES_DIR = Path(__file__).parent


def _ts(day: int, hour: int = 12) -> pd.Timestamp:
    return pd.Timestamp(f"2024-01-{day:02d} {hour:02d}:00:00")


def build_sample_events() -> pd.DataFrame:
    rows = []

    def add(
        user_id,
        day,
        page,
        level="paid",
        length=0.0,
        song=None,
        session=1,
        registration="2023-12-01",
    ):
        rows.append(
            dict(
                userId=user_id,
                time=_ts(day),
                page=page,
                length=length,
                song=song,
                sessionId=session,
                level=level,
                registration=pd.Timestamp(registration),
            )
        )

    # user 1: paid, steady engagement through and after the anchor date -> stays
    for day in [1, 3, 5, 8, 10]:
        add("1", day, "NextSong", length=210.5, song=f"song-{day}", session=day)
    add("1", 9, "Thumbs Up", session=9)

    # user 2: free, engagement drops off, cancels shortly after the anchor -> churns
    for day in [1, 2, 3]:
        add(
            "2",
            day,
            "NextSong",
            level="free",
            length=180.0,
            song=f"song-{day}",
            session=day,
            registration="2024-01-01",
        )
    add("2", 4, "Roll Advert", level="free", session=4, registration="2024-01-01")
    add("2", 15, "Cancel", level="free", session=15, registration="2024-01-01")
    add("2", 15, "Cancellation Confirmation", level="free", session=15, registration="2024-01-01")

    # user 3: paid, tried to downgrade before the anchor, then stays
    for day in [2, 6, 9]:
        add("3", day, "NextSong", length=200.0, song=f"song-{day}", session=day + 20)
    add("3", 6, "Submit Downgrade", session=26)

    # user 4: free, frustrated (errors, ads, thumbs down) but stays
    for day in [1, 4, 7, 9]:
        add(
            "4",
            day,
            "NextSong",
            level="free",
            length=150.0,
            song=f"song-{day}",
            session=day + 40,
            registration="2023-11-01",
        )
    add("4", 7, "Error", level="free", session=47, registration="2023-11-01")
    add("4", 7, "Roll Advert", level="free", session=47, registration="2023-11-01")
    add("4", 9, "Thumbs Down", level="free", session=49, registration="2023-11-01")

    return pd.DataFrame(rows)


def main() -> None:
    events = build_sample_events()
    events.to_csv(FIXTURES_DIR / "sample_events.csv", index=False)

    anchor = _ts(10)
    users = clean_data(build_training_table(events, anchor_date=anchor))
    users.to_csv(FIXTURES_DIR / "sample_users.csv", index=False)

    print(users[["userId", "is_paid", "tenure_days", "target_churn"]])


if __name__ == "__main__":
    main()
