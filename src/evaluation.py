import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return rmse, mae, mape_val

def plot_acf_pacf(series, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(series, ax=axes[0], lags=6)
    axes[0].set_title(f'ACF: {title}')
    plot_pacf(series, ax=axes[1], lags=6, method='ywm')
    axes[1].set_title(f'PACF: {title}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
