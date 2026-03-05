import os

from xgboost import XGBRegressor

import src.config as cfg
from src.loaddata import load_country_data
from src.models.common import (
    expanding_window_forecast,
    grid_dict,
    prepare_train_test,
    resolve_target_column,
    rolling_origin_rmse,
    select_features,
)
from src.preprocess import split_train_test

output_path = os.path.join(cfg.FORECASTS_DIR, "xgb")
os.makedirs(output_path, exist_ok=True)

for country in cfg.COUNTRIES:
    print(f"Processing {country}")
    data = load_country_data("resourceproductivity", [country])[country]
    target_col = resolve_target_column(data, "resourceproductivity")

    train, test = split_train_test(data, train_ratio=cfg.TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test, y_train, y_test = prepare_train_test(train, test, target_col, model_class="tree")

    top_features = select_features(
        X_train,
        y_train,
        threshold=cfg.MUTUAL_INFO_THRESHOLD,
        top_k=cfg.MAX_MI_FEATURES,
    )
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    best_rmse = float("inf")
    best_params = None
    for params in grid_dict(cfg.XGB_PARAMS):
        estimator = XGBRegressor(**params, random_state=42)
        rmse = rolling_origin_rmse(estimator, X_train, y_train, min_train_size=5)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    final_model = XGBRegressor(**best_params, random_state=42)
    forecast_df, train_metrics, test_metrics = expanding_window_forecast(
        estimator=final_model,
        train_df=train,
        test_df=test,
        target_col=target_col,
        model_class="tree",
        selected_features=top_features,
        n_bootstrap=cfg.N_BOOTSTRAP,
    )

    train_rmse, train_mae, train_mape = train_metrics
    rmse, mae, mape_val = test_metrics
    print(
        f"{country} | best={best_params}, CV-RMSE={best_rmse:.3f} "
        f"-> Train RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}, MAPE: {train_mape:.2f}% "
        f"| Test RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape_val:.2f}%"
    )

    forecast_df.to_csv(os.path.join(output_path, f"{country}_xgb_best.csv"))
