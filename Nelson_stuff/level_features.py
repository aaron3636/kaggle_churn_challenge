import pandas as pd

def get_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create level_features identical to the notebook:
    - map {"free": 0, "paid": 1}
    - fillna(0)
    - group by userId and take the max (so if user was ever paid -> 1)
    Returns a DataFrame indexed by userId with column 'level' (int).
    """
    tmp = df[["userId", "level"]].copy()
    level_map = {"free": 0, "paid": 1}
    tmp["level"] = tmp["level"].map(level_map).fillna(0).astype(int)
    level_features = tmp.groupby("userId")["level"].max().to_frame()
    return level_features