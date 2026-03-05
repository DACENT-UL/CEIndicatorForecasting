# Paths
RAW_DATA_DIR = "data/raw/indicator/resourceproductivity"
PROCESSED_DATA_DIR = f"{RAW_DATA_DIR}/processed"
FEATURES_DIR = f"{RAW_DATA_DIR}/features"
TARGET_DIR = f"{RAW_DATA_DIR}/target"
OUTPUT_DIR = f"{RAW_DATA_DIR}/output"

# Output subfolders for specific tasks
CORRELATION_PLOTS_DIR = f"{OUTPUT_DIR}/correlation_plots"
ACF_PLOTS_DIR = f"{OUTPUT_DIR}/acf"
FORECASTS_DIR = f"{OUTPUT_DIR}/forecasts"


# Countries
COUNTRIES = [
    'Belgium', 'Czechia', 'Denmark', 'Germany', 'Estonia',
    'Ireland', 'Greece', 'Spain', 'France', 'Italy', 'Cyprus', 'Latvia',
    'Lithuania', 'Luxembourg', 'Hungary', 'Netherlands', 'Austria',
    'Poland', 'Portugal', 'Romania', 'Slovenia', 'Slovakia', 'Finland', 'Sweden'
]

TRAIN_TEST_SPLIT_RATIO = 0.8
DATE_FORMAT = "%Y"  # For parsing TIME_PERIOD

# Model hyperparameters

# Polynomial features
POLY_DEGREES = [1, 2]

# Ridge regression
RIDGE_ALPHAS = [0.5, 1, 10]

# Lasso regression
LASSO_ALPHAS = [1, 10]

# Random Forest
RF_PARAMS = {
    'n_estimators': [20, 50, 70],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# XGBoost
XGB_PARAMS = {
    'n_estimators': [20, 50, 70],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.02, 0.05],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8]
}

# SVR
SVR_PARAMS = {
    'C': [0.01, 0.1, 1],
    'epsilon': [0.1, 0.5, 1]
}

# Bootstrap
N_BOOTSTRAP = 1000


# Feature selection thresholds
MUTUAL_INFO_THRESHOLD = 0.01
MAX_MI_FEATURES = 10
HIGH_CORR_THRESHOLD = 0.9  # For dropping features with high correlation
