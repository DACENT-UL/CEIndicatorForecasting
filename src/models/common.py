from itertools import product
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from src.evaluation import compute_metrics
from src.featureselection import mutual_info_ranking
from src.preprocess import check_stationarity


def _stationarity_flags(train_df: pd.DataFrame) -> Dict[str, bool | None]:
    return {col: check_stationarity(train_df[col]) for col in train_df.columns}


def _difference_with_reference(train_col: pd.Series, test_col: pd.Series) -> Tuple[pd.Series, pd.Series]:
    train_diff = train_col.diff().dropna()
    combined = pd.concat([train_col.iloc[[-1]], test_col])
    test_diff = combined.diff().iloc[1:]
    test_diff.index = test_col.index
    return train_diff, test_diff


def prepare_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    model_class: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare train/test according to model class assumptions.

    model_class='linear': keep levels and add time_index feature.
    model_class='tree': difference non-stationary columns; keep stationary columns.
    """
    if model_class == "linear":
        train = train_df.copy()
        test = test_df.copy()
        train["time_index"] = np.arange(len(train), dtype=float)
        test["time_index"] = np.arange(len(train), len(train) + len(test), dtype=float)
    elif model_class == "tree":
        flags = _stationarity_flags(train_df)
        train = pd.DataFrame(index=train_df.index)
        test = pd.DataFrame(index=test_df.index)

        for col, is_stat in flags.items():
            if is_stat is None:
                continue
            if is_stat:
                train[col] = train_df[col]
                test[col] = test_df[col]
            else:
                tr, te = _difference_with_reference(train_df[col], test_df[col])
                train[col] = tr
                test[col] = te

        common_cols = [c for c in train.columns if c in test.columns]
        train = train[common_cols].dropna()
        test = test[common_cols].dropna()
    else:
        raise ValueError(f"Unknown model_class: {model_class}")

    train = train.dropna()
    test = test.dropna()

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]
    return X_train, X_test, y_train, y_test


def select_features(X_train: pd.DataFrame, y_train: pd.Series, threshold: float, top_k: int = 10) -> List[str]:
    mi_scores = mutual_info_ranking(X_train, y_train)
    top = mi_scores[mi_scores >= threshold].head(top_k).index.tolist()
    if not top:
        top = mi_scores.head(top_k).index.tolist()
    return top


def rolling_origin_rmse(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    min_train_size: int = 5,
) -> float:
    if len(X) <= min_train_size:
        return float("inf")

    preds: List[float] = []
    truths: List[float] = []

    for t in range(min_train_size, len(X)):
        X_tr = X.iloc[:t]
        y_tr = y.iloc[:t]
        X_val = X.iloc[t : t + 1]
        y_val = y.iloc[t : t + 1]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        model = clone(estimator)
        model.fit(X_tr_scaled, y_tr)
        pred = model.predict(X_val_scaled)

        preds.append(float(pred[0]))
        truths.append(float(y_val.iloc[0]))

    rmse, _, _ = compute_metrics(truths, preds)
    return float(rmse)


def grid_dict(param_grid: Dict[str, Iterable]) -> Iterable[Dict[str, object]]:
    keys = list(param_grid.keys())
    for values in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, values))


def resolve_target_column(df: pd.DataFrame, indicator: str) -> str:
    expected = indicator.replace("_", "").lower()
    for col in df.columns:
        if col.replace("_", "").lower() == expected:
            return col
    raise KeyError(f"Target column matching {indicator!r} not found in columns: {list(df.columns)}")


def expanding_window_forecast(
    estimator,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    model_class: str,
    selected_features: List[str],
    n_bootstrap: int = 1000,
) -> Tuple[pd.DataFrame, Tuple[float, float, float], Tuple[float, float, float]]:
    """Run one-step-ahead expanding-window forecast with bootstrap prediction intervals."""
    train_window = train_df.copy()
    train_fitted_preds: List[float] = []
    train_true: List[float] = []
    predictions: List[float] = []
    true_values: List[float] = []
    residuals: List[float] = []
    bootstrap_rows: List[np.ndarray] = []
    forecast_index: List[object] = []

    for i in range(len(test_df)):
        test_step = test_df.iloc[i : i + 1]
        X_train, X_test, y_train, y_test = prepare_train_test(train_window, test_step, target_col, model_class=model_class)

        X_train_sel = X_train.reindex(columns=selected_features, fill_value=0.0)
        X_test_sel = X_test.reindex(columns=selected_features, fill_value=0.0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        model = clone(estimator)
        model.fit(X_train_scaled, y_train)

        fitted_train = model.predict(X_train_scaled)
        train_fitted_preds.extend(np.asarray(fitted_train, dtype=float).tolist())
        train_true.extend(np.asarray(y_train, dtype=float).tolist())

        y_pred = float(model.predict(X_test_scaled)[0])
        y_true = float(y_test.iloc[0])
        predictions.append(y_pred)
        true_values.append(y_true)
        forecast_index.append(y_test.index[0])

        train_resid_std = float(np.std(np.asarray(y_train) - np.asarray(fitted_train)))
        resid_std = train_resid_std if i == 0 else float(np.std(residuals))
        y_boot = y_pred + np.random.normal(0, resid_std, size=n_bootstrap)
        bootstrap_rows.append(y_boot)

        residuals.append(y_true - y_pred)
        train_window = pd.concat([train_window, test_step])

    bootstrap_array = np.vstack(bootstrap_rows)
    forecast_df = pd.DataFrame(
        {
            "True": true_values,
            "Predicted": predictions,
            "Mean_Bootstrap": bootstrap_array.mean(axis=1),
            "Lower95_CI": np.percentile(bootstrap_array, 2.5, axis=1),
            "Upper95_CI": np.percentile(bootstrap_array, 97.5, axis=1),
        },
        index=forecast_index,
    )

    train_metrics = compute_metrics(train_true, train_fitted_preds)
    test_metrics = compute_metrics(forecast_df["True"], forecast_df["Predicted"])
    return forecast_df, train_metrics, test_metrics
