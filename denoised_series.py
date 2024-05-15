import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pywt

# Function to denoise a signal using wavelet transform
def denoise_signal(signal, wavelet='db1',level =1):
    # Découper la transformée en ondelettes
    coeffs = pywt.wavedec(signal, wavelet)
    
    def madev(d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    # Calculer le seuil universel
    sigma = madev(coeffs[-level])/ 0.6745
    #sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Appliquer le seuil
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    
    # Reconstruire le signal débruité
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)

    return denoised_signal


def denoise_all_signal(df_price, wavelet='db1',level =1):
    return_df = df_price.pct_change(1).fillna(0)

    # Appliquer la fonction de débruitage
    denoised_series = {}
    for column in return_df.columns:
        signal = return_df[column].values
        denoised_signal = denoise_signal(signal, wavelet=wavelet,level = level)  # Ajuster le niveau si nécessaire
        denoised_series[column] = denoised_signal[:len(return_df)]  # Ajuster à la longueur originale si nécessaire

    # Creating a dataframe for the denoised series
    denoised_return_df = pd.DataFrame(denoised_series, index=return_df.index)

    # Utiliser les prix initiaux réels
    initial_prices = df_price.iloc[0]
    # Calculer les prix en une seule ligne en utilisant les prix initiaux de la première journée
    denoised_price_df = pd.DataFrame({asset: initial_prices[asset] * (1 + denoised_return_df[asset]).cumprod() for asset in denoised_return_df.columns})

    return denoised_return_df, denoised_price_df

def plot_comparison(initial_df, denoised_df):

    # Déterminer le nombre de colonnes pour configurer les subplots
    num_columns = len(initial_df.columns)
    plt.figure(figsize=(15, 5 * num_columns))  # Ajuster la taille du graphique en fonction du nombre de colonnes

    for i, column in enumerate(initial_df.columns, 1):
        # Tracer la série originale
        plt.subplot(num_columns, 2, 2*i-1)
        plt.plot(initial_df.index, initial_df[column], label="Original", color='blue')
        plt.title(f"Original {column}")
        plt.legend()

        # Tracer la série débruitée
        plt.subplot(num_columns, 2, 2*i)
        plt.plot(denoised_df.index, denoised_df[column], label="Denoised", color='green')
        plt.title(f"Denoised {column}")
        plt.legend()

    plt.tight_layout()  # Ajuster automatiquement le layout
    plt.show()