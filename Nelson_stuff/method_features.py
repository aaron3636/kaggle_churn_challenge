import pandas as pd
import numpy as np

def get_method_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create method_features identical to the notebook:
    - map {"PUT":0, "GET":1}
    - fillna(-1)
    - group by userId and take the max method value per user
    Returns a DataFrame indexed by userId with column 'method' (int).
    """
    tmp = df[["userId", "method"]].copy()
    method_map = {"PUT": 0, "GET": 1}
    tmp["method"] = tmp["method"].map(method_map).fillna(-1).astype(int)
    method_features = tmp.groupby("userId")["method"].max().to_frame()
    return method_features