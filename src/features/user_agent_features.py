import pandas as pd


def device_type(user_agent: str) -> str:
    """
    Map a user agent string to a coarse device label.
    Returns "Different Device" when the user agent is missing or unrecognized.
    """
    if pd.isna(user_agent):
        return "Different Device"

    user_agent = str(user_agent)

    if "Windows" in user_agent:
        return "Windows"
    if "Macintosh" in user_agent:
        return "Macintosh"
    if "Linux" in user_agent:
        return "Linux"
    if "iPad" in user_agent:
        return "iPad"
    if "iPhone" in user_agent:
        return "iPhone"
    return "Different Device"


def device_used_by_user(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["userId", "userAgent"]].drop_duplicates().copy()

    df.loc[:, "device_used"] = df["userAgent"].str.extract(r"(\([^)]*\))", expand=False)
    df.loc[:, "exact_device"] = df["device_used"].apply(device_type)
    return df[["userId", "exact_device"]]
