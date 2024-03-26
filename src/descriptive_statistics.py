import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller

def descriptive_statistics(serie):
    # Statistiques pour la série originale
    original_stats = {
        'Mean': np.mean(serie),
        'Median': np.median(serie),
        'Minimum': np.min(serie),
        'Maximum': np.max(serie),
        'Std deviation': np.std(serie, ddof=1),
        'Skewness': skew(serie),
        'Kurtosis': kurtosis(serie),
        'Autocorrelation': pd.Series(serie).autocorr(),
        'ADF test p-value (10 lags)': adfuller(serie, maxlag=10)[1],
        'Nb obs': len(serie)
    }

    # Calcul des rendements logarithmiques de la série
    if len(serie) > 1:
        log_returns = np.diff(np.log(serie))
    else:
        log_returns = []

    # Vérifier si log_returns contient des données avant de calculer les statistiques
    log_stats = {
        'Log Mean': np.mean(log_returns),
        'Log Median': np.median(log_returns),
        'Log Minimum': np.min(log_returns),
        'Log Maximum': np.max(log_returns),
        'Log Std deviation': np.std(log_returns, ddof=1),
        'Log Skewness': skew(log_returns),
        'Log Kurtosis': kurtosis(log_returns),
        'Log Autocorrelation': pd.Series(log_returns).autocorr(),
        'Log ADF test p-value (10 lags)': adfuller(log_returns, maxlag=10)[1],
        'Log Nb obs': len(log_returns)
    }


    return original_stats, log_stats

