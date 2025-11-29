import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from features import feature_total_pages_count

df_train = pd.read_parquet("data/churn-prediction-25-26/train.parquet")
page_features = feature_total_pages_count(df_train)


print(page_features)


columns_train =
