import pandas as pd
import datetime
# needs pyarrow or fastparquet installed
df_train = pd.read_parquet("1_Data/churn-prediction-25-26/train.parquet")
# df_test = pd.read_parquet("1_Data/churn-prediction-25-26/test.parquet")

print(df_train.info())
# print(df_test.info())

# print(4393179 / (17499636 + 4393179))
# print(17499636 + 4393179)


# print(pd.to_datetime(1542671096000, unit="ms", utc=True))              # UTC
# print(pd.to_datetime(1542671174000, unit="ms", utc=True))              # UTC

# print(pd.to_datetime(1542671096000, unit="ms", utc=True).tz_convert("Europe/Paris"))


print(df_train["page"].unique())


# Page option
['NextSong' 'Downgrade' 'Help' 'Home' 'Thumbs Up' 'Add Friend'
 'Thumbs Down' 'Add to Playlist' 'Logout' 'About' 'Settings'
 'Save Settings' 'Cancel' 'Cancellation Confirmation' 'Submit Downgrade'
 'Roll Advert' 'Upgrade' 'Error' 'Submit Upgrade']


print(df_train[df_train["userId"] == "1152881"])
