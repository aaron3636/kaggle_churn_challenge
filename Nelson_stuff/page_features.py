import pandas as pd

def create_page_features(df):
    return pd.crosstab(df["userId"], df["page"])
