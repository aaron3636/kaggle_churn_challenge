import pandas as pd
import datetime

# MODEL TRAIN TEST WINDOWS
train_test_bins_1_week_window = [
    # only window with 8 days in the train set
    (datetime.date(2018, 10, 1), datetime.date(2018, 10, 8),
     datetime.date(2018, 10, 19), "01/10/2018–08/10/2018"),
    (datetime.date(2018, 10, 9), datetime.date(2018, 10, 15),
     datetime.date(2018, 10, 26), "09/10/2018–15/10/2018"),
    (datetime.date(2018, 10, 16), datetime.date(2018, 10, 22),
     datetime.date(2018, 11, 2), "16/10/2018–22/10/2018"),
    (datetime.date(2018, 10, 23), datetime.date(2018, 10, 29),
     datetime.date(2018, 11, 9), "23/10/2018–29/10/2018"),
    (datetime.date(2018, 10, 30), datetime.date(2018, 11, 5),
     datetime.date(2018, 11, 16), "30/10/2018–05/11/2018"),
    # only 7 day furture churn window in the data available
    (datetime.date(2018, 11, 6), datetime.date(2018, 11, 12),
     datetime.date(2018, 11, 30), "06/11/2018–12/11/2018")
]

train_test_bins_2_week_window = [
    (datetime.date(2018, 10, 1), datetime.date(2018, 10, 15),
     datetime.date(2018, 10, 26), "01/10/2018–15/10/2018"),
    (datetime.date(2018, 10, 16), datetime.date(2018, 10, 29),
     datetime.date(2018, 11, 9), "16/10/2018–29/10/2018"),
    # only 7 day furture churn window in the data available
    (datetime.date(2018, 10, 30), datetime.date(2018, 11, 12),
     datetime.date(2018, 11, 30), "30/10/2018–12/11/2018")
]

train_test_bins_3_week_window = [
    (datetime.date(2018, 10, 1), datetime.date(2018, 10, 22),
     datetime.date(2018, 11, 2), "01/10/2018–22/10/2018"),
    # only 7 day furture churn window in the data available
    (datetime.date(2018, 10, 23), datetime.date(2018, 11, 12),
     datetime.date(2018, 11, 30), "23/10/2018–12/11/2018")
]


# SUBMISSION WINDOWS
submission_1_week_window = [
    # only window with 8 days in the train set
    (datetime.date(2018, 10, 1), datetime.date(2018, 10, 8), "01/10/2018–08/10/2018"),
    (datetime.date(2018, 10, 9), datetime.date(2018, 10, 15), "09/10/2018–15/10/2018"),
    (datetime.date(2018, 10, 16), datetime.date(2018, 10, 22), "16/10/2018–22/10/2018"),
    (datetime.date(2018, 10, 23), datetime.date(2018, 10, 29), "23/10/2018–29/10/2018"),
    (datetime.date(2018, 10, 30), datetime.date(2018, 11, 5), "30/10/2018–05/11/2018"),
    (datetime.date(2018, 11, 6), datetime.date(2018, 11, 12), "06/11/2018–12/11/2018"),
    (datetime.date(2018, 11, 13), datetime.date(2018, 11, 19), "13/11/2018–19/11/2018"),
]

submission_2_week_window = [
    (datetime.date(2018, 10, 9), datetime.date(2018, 10, 22), "09/10/2018–22/10/2018"),
    (datetime.date(2018, 10, 23), datetime.date(2018, 11, 5), "23/10/2018–05/11/2018"),
    (datetime.date(2018, 11, 6), datetime.date(2018, 11, 19), "06/11/2018–19/11/2018")
]

submission_3_week_window = [
    (datetime.date(2018, 10, 9), datetime.date(2018, 10, 29), "09/10/2018–29/10/2018"),
    (datetime.date(2018, 10, 30), datetime.date(2018, 11, 19), "30/10/2018–19/11/2018")
]


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


def create_submission_time_window(df: pd.DataFrame,
                                  t0: datetime.date,
                                  t1: datetime.date,
                                  week_slice: int):

    df = df.copy()
    df["date"] = pd.to_datetime(df["time"]).dt.date

    present = df[(df["date"] >= t0) & (df["date"] <= t1)]

    # present["week_slice"] = week_slice

    return present


def creat_X_y_time_window(df: pd.DataFrame,
                          t0: datetime.date,
                          t1: datetime.date,
                          t2: datetime.date):

    df = df.copy()
    df["date"] = pd.to_datetime(df["time"]).dt.date

    past = df[df["date"] < t0]
    present = df[(df["date"] >= t0) & (df["date"] <= t1)]

    canceled_before = past.loc[past["page"].eq(
        "Cancellation Confirmation"), "userId"].unique()
    canceled_present = present.loc[present["page"].eq(
        "Cancellation Confirmation"), "userId"].unique()

    present_active = present[(~present["userId"].isin(canceled_before)) &
                             (~present["userId"].isin(canceled_present))].copy()

    cohort = present_active[["userId"]].drop_duplicates()

    # skip entire next day
    gap_day = t1 + datetime.timedelta(days=1)
    canceled_gap = df.loc[(df["date"] == gap_day) &
                          (df["page"].eq("Cancellation Confirmation")), "userId"].unique()

    cohort = cohort[~cohort["userId"].isin(canceled_gap)]
    present_active = present_active[present_active["userId"].isin(cohort["userId"])]

    # present_active["week_slice"] = week_slice

    # --- y: cancellation in (t1+1 day, t2] -> starts t1+2 days because of ">"
    future = df[(df["date"] > gap_day) & (df["date"] <= t2)]
    future = future[future["userId"].isin(cohort["userId"])]

    y = (future["page"].eq("Cancellation Confirmation")
         .groupby(future["userId"]).max().astype("int8")
         .rename("Cancellation Confirmation")
         .reset_index())

    y = (cohort.merge(y, on="userId", how="left")
         .fillna({"Cancellation Confirmation": 0})
         .astype({"Cancellation Confirmation": "int8"}))
    # y["week_slice"] = week_slice

    return present_active, y

# MODEL TRAIN DATA CREATION FUNCTIONS


def create_1_week_X_y_df(df: pd.DataFrame):
    time_frame_slices_1_week = {}

    for t0, t1, t2, slice_label in train_test_bins_1_week_window:
        time_frame_slices_1_week[slice_label] = creat_X_y_time_window(df=df,
                                                                      t0=t0,
                                                                      t1=t1,
                                                                      t2=t2)

    df_X_week_1, df_y_week_1 = time_frame_slices_1_week["01/10/2018–08/10/2018"]
    df_X_week_2, df_y_week_2 = time_frame_slices_1_week["09/10/2018–15/10/2018"]
    df_X_week_3, df_y_week_3 = time_frame_slices_1_week["16/10/2018–22/10/2018"]
    df_X_week_4, df_y_week_4 = time_frame_slices_1_week["23/10/2018–29/10/2018"]
    df_X_week_5, df_y_week_5 = time_frame_slices_1_week["30/10/2018–05/11/2018"]
    df_X_week_6, df_y_week_6 = time_frame_slices_1_week["06/11/2018–12/11/2018"]

    """
    df_X_all_week_slices = pd.concat([df_X_week_1,
                                      df_X_week_2,
                                      df_X_week_3,
                                      df_X_week_4,
                                      df_X_week_5,
                                      df_X_week_6], axis=0)

    df_y_all_week_slices = pd.concat([df_y_week_1,
                                      df_y_week_2,
                                      df_y_week_3,
                                      df_y_week_4,
                                      df_y_week_5,
                                      df_y_week_6], axis=0)

    return df_X_all_week_slices, df_y_all_week_slices
    """

    df_X_week_slices = [df_X_week_1, df_X_week_2, df_X_week_3, df_X_week_4,
                        df_X_week_5, df_X_week_6]

    df_y_week_slices = [df_y_week_1, df_y_week_2, df_y_week_3, df_y_week_4,
                        df_y_week_5, df_y_week_6]

    return df_X_week_slices, df_y_week_slices


def create_2_week_X_y_df(df: pd.DataFrame):
    time_frame_slices_2_week = {}

    for t0, t1, t2, slice_label in train_test_bins_2_week_window:
        time_frame_slices_2_week[slice_label] = creat_X_y_time_window(df=df,
                                                                      t0=t0,
                                                                      t1=t1,
                                                                      t2=t2)

    df_X_week_1, df_y_week_1 = time_frame_slices_2_week["01/10/2018–15/10/2018"]
    df_X_week_2, df_y_week_2 = time_frame_slices_2_week["16/10/2018–29/10/2018"]
    df_X_week_3, df_y_week_3 = time_frame_slices_2_week["30/10/2018–12/11/2018"]
    """
    df_X_all_week_slices = pd.concat([df_X_week_1,
                                      df_X_week_2,
                                      df_X_week_3], axis=0)

    df_y_all_week_slices = pd.concat([df_y_week_1,
                                      df_y_week_2,
                                      df_y_week_3], axis=0)

    return df_X_all_week_slices, df_y_all_week_slices
    """

    df_X_week_slices = [df_X_week_1, df_X_week_2, df_X_week_3]

    df_y_week_slices = [df_y_week_1, df_y_week_2, df_y_week_3]

    return df_X_week_slices, df_y_week_slices


def create_3_week_X_y_df(df: pd.DataFrame):
    time_frame_slices_3_week = {}

    for t0, t1, t2, slice_label in train_test_bins_3_week_window:
        time_frame_slices_3_week[slice_label] = creat_X_y_time_window(df=df,
                                                                      t0=t0,
                                                                      t1=t1,
                                                                      t2=t2)

    df_X_week_1, df_y_week_1 = time_frame_slices_3_week["01/10/2018–22/10/2018"]
    df_X_week_2, df_y_week_2 = time_frame_slices_3_week["23/10/2018–12/11/2018"]
    """
    df_X_all_week_slices = pd.concat([df_X_week_1,
                                      df_X_week_2], axis=0)

    df_y_all_week_slices = pd.concat([df_y_week_1,
                                      df_y_week_2], axis=0)

    return df_X_all_week_slices, df_y_all_week_slices
    """

    df_X_week_slices = [df_X_week_1, df_X_week_2]

    df_y_week_slices = [df_y_week_1, df_y_week_2]

    return df_X_week_slices, df_y_week_slices


# SUBMISSION TIME WINDOW CREATION FUNCTIONS

def create_1_week_X_y_df_test(df: pd.DataFrame):
    time_frame_slices_1_week = {}

    for week_slice, (t0, t1, slice_label) in enumerate(submission_1_week_window, start=1):
        time_frame_slices_1_week[slice_label] = create_submission_time_window(df=df,
                                                                              t0=t0,
                                                                              t1=t1,
                                                                              week_slice=week_slice)

    df_X_week_1 = time_frame_slices_1_week[submission_1_week_window[0][2]]
    df_X_week_2 = time_frame_slices_1_week[submission_1_week_window[1][2]]
    df_X_week_3 = time_frame_slices_1_week[submission_1_week_window[2][2]]
    df_X_week_4 = time_frame_slices_1_week[submission_1_week_window[3][2]]
    df_X_week_5 = time_frame_slices_1_week[submission_1_week_window[4][2]]
    df_X_week_6 = time_frame_slices_1_week[submission_1_week_window[5][2]]
    df_X_week_7 = time_frame_slices_1_week[submission_1_week_window[6][2]]

    df_X_all_week_slices = pd.concat([df_X_week_1,
                                      df_X_week_2,
                                      df_X_week_3,
                                      df_X_week_4,
                                      df_X_week_5,
                                      df_X_week_6,
                                      df_X_week_7], axis=0)

    return df_X_all_week_slices


def create_2_week_X_y_df_test(df: pd.DataFrame):
    time_frame_slices_2_week = {}

    for week_slice, (t0, t1, slice_label) in enumerate(submission_2_week_window, start=1):
        time_frame_slices_2_week[slice_label] = create_submission_time_window(df=df,
                                                                              t0=t0,
                                                                              t1=t1,
                                                                              week_slice=week_slice)

    df_X_week_1 = time_frame_slices_2_week[submission_2_week_window[0][2]]
    df_X_week_2 = time_frame_slices_2_week[submission_2_week_window[1][2]]
    df_X_week_3 = time_frame_slices_2_week[submission_2_week_window[2][2]]

    df_X_all_week_slices = pd.concat([df_X_week_1,
                                      df_X_week_2,
                                      df_X_week_3], axis=0)

    return df_X_all_week_slices


def create_3_week_X_y_df_test(df: pd.DataFrame):
    time_frame_slices_3_week = {}

    for week_slice, (t0, t1, slice_label) in enumerate(submission_3_week_window, start=1):
        time_frame_slices_3_week[slice_label] = create_submission_time_window(df=df,
                                                                              t0=t0,
                                                                              t1=t1,
                                                                              week_slice=week_slice)

    df_X_week_1 = time_frame_slices_3_week[submission_3_week_window[0][2]]
    df_X_week_2 = time_frame_slices_3_week[submission_3_week_window[1][2]]

    df_X_all_week_slices = pd.concat([df_X_week_1,
                                      df_X_week_2], axis=0)

    return df_X_all_week_slices
