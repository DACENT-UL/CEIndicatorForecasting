import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def clean_features(train_df, target_col, threshold_corr=0.9):
    corr_matrix = train_df.corr()
    features = [c for c in train_df.columns if c != target_col]
    target_corr = corr_matrix.loc[features, target_col].abs()
    feature_corr = corr_matrix.loc[features, features].abs()
    upper_tri = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper_tri.columns:
        for row in upper_tri.index:
            if pd.isna(upper_tri.loc[row, col]):
                continue
            if upper_tri.loc[row, col] >= threshold_corr:
                if target_corr[row] < target_corr[col]:
                    to_drop.add(row)
                else:
                    to_drop.add(col)

    cleaned_train = train_df.drop(columns=list(to_drop))
    return cleaned_train, to_drop

def mutual_info_ranking(X, y):
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return mi_series
