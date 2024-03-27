import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller

def calculate_statistics(serie, prefix=''):
    return {
        f'{prefix}Mean': np.mean(serie),
        f'{prefix}Median': np.median(serie),
        f'{prefix}Minimum': np.min(serie),
        f'{prefix}Maximum': np.max(serie),
        f'{prefix}Std deviation': np.std(serie, ddof=1),
        f'{prefix}Skewness': skew(serie),
        f'{prefix}Kurtosis': kurtosis(serie),
        f'{prefix}Autocorrelation': pd.Series(serie).autocorr(),
        f'{prefix}ADF test p-value (10 lags)': adfuller(serie, maxlag=10)[1],
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

