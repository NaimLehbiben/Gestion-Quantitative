import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller

def calculate_statistics(serie, prefix=''):
    """
    Calculate various statistics for a given series.

    Parameters:
    serie (array-like): The data series to calculate statistics for.
    prefix (str): A prefix to add to each statistic's name.

    Returns:
    dict: A dictionary containing the calculated statistics.
    """
    return {
        f'{prefix}Mean': round(np.mean(serie), 4),  # Mean of the series
        f'{prefix}Median': round(np.median(serie), 4),  # Median of the series
        f'{prefix}Minimum': round(np.min(serie), 3),  # Minimum value in the series
        f'{prefix}Maximum': round(np.max(serie), 3),  # Maximum value in the series
        f'{prefix}Std deviation': round(np.std(serie, ddof=1), 4),  # Standard deviation of the series
        f'{prefix}Skewness': round(skew(serie), 2),  # Skewness of the series
        f'{prefix}Kurtosis': round(kurtosis(serie), 2),  # Kurtosis of the series
        f'{prefix}Autocorrelation': round(pd.Series(serie).autocorr(), 3),  # Autocorrelation of the series
        f'{prefix}ADF test p-value (10 lags)': round(adfuller(serie, maxlag=10)[1], 2),  # p-value from ADF test with 10 lags
        f'{prefix}Nb obs': len(serie)  # Number of observations in the series
    }

def descriptive_statistics(serie):
    """
    Calculate and return descriptive statistics for a given series and its log returns.

    Parameters:
    serie (array-like): The data series to calculate statistics for.

    Returns:
    tuple: A tuple containing two dictionaries, one for the original series and one for the log returns.
    """
    # Calculate statistics for the original series
    original_stats = calculate_statistics(serie)

    # Calculate log returns if the series has more than one element
    if len(serie) > 1:
        log_returns = np.diff(np.log(serie))
    else:
        log_returns = []

    # Calculate statistics for the log returns
    log_stats = calculate_statistics(log_returns, 'Log ')

    return original_stats, log_stats
