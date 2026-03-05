# CEIndicatorForecasting

Forecasting framework for EU Circular Economy Indicators. Data included in this repository is sourced from the Eurostat database: https://ec.europa.eu/eurostat/web/main/data/database 

## Local setup

### 1) Clone and enter the project
```bash
git clone https://github.com/DACENT-UL/CEIndicatorForecasting
cd CEIndicatorForecasting
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

## Project structure (relevant parts)

- `src/train_models.py`: CLI entrypoint to run one model or all models.
- `src/models/*.py`: per-model training pipelines.
- `src/loaddata.py`: loader that reads `data/raw/indicator/<indicator>/features` and `.../target`.
- `src/config.py`: countries, split ratio, hyperparameters, no. of features, output paths.

## How to run training locally

From the repo root:

### Run a single model
```bash
python -m src.train_models --model ridge
```

Valid model values:
- `ridge`
- `lasso`
- `rf`
- `xgb`
- `svr`

### Run all models
```bash
python -m src.train_models --model all
```


## Where outputs are written

Forecast CSVs are written under:

`data/raw/indicator/resourceproductivity/output/forecasts/<model_name>/`

(Example: `.../forecasts/poly_ridge/` for ridge outputs.)

## Notes for local runs

- Ensure your raw data folders exist for the target indicator:
  - `data/raw/indicator/resourceproductivity/features`
  - `data/raw/indicator/resourceproductivity/target`
- The current model scripts are configured to load `resourceproductivity`.
- If you want a different indicator, update the indicator argument in each model script call to `load_country_data(...)`.


## Preprocessing and model selection

- **Linear models (Ridge, Lasso, SVR):** use level data (no differencing) and append a `time_index` feature so models can learn trend.
- **Tree-based models (Random Forest, XGBoost):** apply first-order differencing only for columns that are non-stationary (ADF test on train split), while stationary columns are kept in levels.
- **Scaling:** a `StandardScaler` is fit on each training window and applied to the corresponding validation/test window.
- **Hyperparameter selection:** grid search is evaluated with **one-step-ahead rolling-origin RMSE**; the best config is then fit once on train and evaluated on the holdout test split.


#







