import pandas as pd


# Second Train Test set
cutoff_train = pd.Timestamp("2018-10-15")
cutoff_test = pd.Timestamp("2018-10-25")


def create_X_y_df_1(df: pd.DataFrame,
                    cut_off_date_train,
                    cut_off_date_test):
    """
    Creating a time series train test split has two problems I have so far
    discovered:

    General procedure:

    Define train cut off date
    Define test cut off date: on our case 10 days into the future

    Create a train dataset that includes all user activity up until the train cut off point
    Create a test dataset that includes all user activity up until the test cut off point

    Now, in the train datset we also have users activity that have already
    canceled their memebership, we do not care about them, as we are trying
    to predict the future here for our current still active users. So
    we remove them for the train dataset.

    For the test dataset we only consider the users that are also in the train
    set so we remove all newly joined users and create a taret column that
    shows, whether they have canceled their subscription or not.

    """
    df["date"] = pd.to_datetime(df["time"])

    df_train = df.copy()
    df_test = df.copy()

    df_train = df_train[df_train["date"] <= cut_off_date_train].copy()
    df_test = df_test[df_test["date"] <= cut_off_date_test].copy()

    # only consider users that haven't already cancelled their memebership
    df_train["already_canceled"] = df_train["page"] == "Cancellation Confirmation"

    # only consider users that are still active
    df_train_still_active = df_train[~df_train["userId"].isin(
        df_train[df_train["already_canceled"]]["userId"])].copy()

    # only consider users that where in the pre train cut off
    # we do not consider new users
    df_test_reduced = df_test[df_test["userId"].isin(
        df_train_still_active["userId"])].copy()

    # creates the target labeling
    df_test_reduced["Canceled"] = df_test_reduced["page"] == "Cancellation Confirmation"
    df_test_reduced["Canceled"] = df_test_reduced["Canceled"].astype(int)

    # take the max Canceled, which if canceled should be equal to 1
    df_test_reduced["Cancellation Confirmation"] = df_test_reduced.groupby(
        "userId")["Canceled"].transform("max")

    df_y = df_test_reduced[["userId", "Cancellation Confirmation"]].drop_duplicates()

    return df_train_still_active, df_y


def create_X_y_df(df: pd.DataFrame, t0, t1):
    df = df.copy()
    df["date"] = pd.to_datetime(df["time"])
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)

    past = df[df["date"] <= t0].copy()

    # cohort: active at t0 (no cancellation on/before t0)
    canceled_before = past.loc[past["page"].eq(
        "Cancellation Confirmation"), "userId"].unique()
    past_active = past[~past["userId"].isin(canceled_before)].copy()
    cohort = past_active[["userId"]].drop_duplicates()

    # labels: cancellation in (t0, t1]
    future = df[(df["date"] > t0) & (df["date"] <= t1)]
    future = future[future["userId"].isin(cohort["userId"])]

    y = (future["page"].eq("Cancellation Confirmation")
         .groupby(future["userId"]).max().astype("int8")
         .rename("Cancellation Confirmation")
         .reset_index())

    y = (cohort.merge(y, on="userId", how="left")
               .fillna({"Cancellation Confirmation": 0})
               .astype({"Cancellation Confirmation": "int8"}))

    return past_active, y
