import pandas as pd

def feature_selection(df):
    df = df.copy()
    if "Transaction Date and Time" in df.columns:
        df["Transaction Date and Time"] = pd.to_datetime(df["Transaction Date and Time"], errors="coerce")
        df["Hour"] = df["Transaction Date and Time"].dt.hour
        df["DayOfWeek"] = df["Transaction Date and Time"].dt.dayofweek
        df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
        def get_time_of_day(hour):
            if pd.isnull(hour):
                return None
            elif 6 <= hour < 18:
                return "Morning"
            else:
                return "Night"
        df["TimeOfDay"] = df["Hour"].apply(get_time_of_day)
    return df
