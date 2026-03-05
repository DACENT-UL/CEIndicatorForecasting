import pandas as pd
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    series = series.dropna()
    if series.nunique() <= 1:
        return None
    result = adfuller(series)
    return result[1] < 0.05  # True = stationary

def split_train_test(df, train_ratio=0.8):
    n = len(df)
    train = df.iloc[:int(n*train_ratio)]
    test = df.iloc[int(n*train_ratio):]
    return train, test

def difference_nonstationary(df):
    diff_df = pd.DataFrame(index=df.index[1:])
    for col in df.columns:
        is_stat = check_stationarity(df[col])
        if is_stat is None:
            continue
        elif not is_stat:
            diff_df[col] = df[col].diff().dropna()
        else:
            diff_df[col] = df[col][1:]
    return diff_df
