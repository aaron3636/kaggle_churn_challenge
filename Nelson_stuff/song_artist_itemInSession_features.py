import pandas as pd

def create_df_autres(df):
    return df.groupby('userId').agg({
        "song": 'nunique',
        "artist": 'nunique',
        "itemInSession": 'max',
    })