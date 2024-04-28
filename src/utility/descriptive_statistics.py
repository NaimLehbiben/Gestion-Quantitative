import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller

def calculate_statistics(serie, prefix=''):
    return {
        f'{prefix}Mean': round(np.mean(serie), 4),
        f'{prefix}Median': round(np.median(serie), 4),
        f'{prefix}Minimum': round(np.min(serie), 3),
        f'{prefix}Maximum': round(np.max(serie), 3),
        f'{prefix}Std deviation': round(np.std(serie, ddof=1), 4),
        f'{prefix}Skewness': round(skew(serie), 2),
        f'{prefix}Kurtosis': round(kurtosis(serie), 2),
        f'{prefix}Autocorrelation': round(pd.Series(serie).autocorr(), 3),
        f'{prefix}ADF test p-value (10 lags)': round(adfuller(serie, maxlag=10)[1], 2),
        f'{prefix}Nb obs': len(serie)
    }

def descriptive_statistics(serie):
    original_stats = calculate_statistics(serie)

    if len(serie) > 1:
        log_returns = np.diff(np.log(serie))
    else:
        log_returns = []

    log_stats = calculate_statistics(log_returns, 'Log ')

    return original_stats, log_stats