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

    # Appliquer la fonction de débruitage
    denoised_price = {}
    for column in df_price.columns:
        signal = df_price[column].values
        denoised_signal = denoise_signal(signal, wavelet=wavelet,level = level)  # Ajuster le niveau si nécessaire
        denoised_price[column] = denoised_signal[:len(df_price)]  # Ajuster à la longueur originale si nécessaire

    # Creating a dataframe for the denoised series
    denoised_price_df = pd.DataFrame(denoised_price, index=df_price.index)

    return denoised_price_df


def plot_comparaison(initial_df, denoised_df):
    # Déterminer le nombre de colonnes pour configurer les subplots
    num_columns = len(initial_df.columns)
    plt.figure(figsize=(15, 5 * num_columns))  # Ajuster la taille du graphique en fonction du nombre de colonnes

    for i, column in enumerate(initial_df.columns, start=1):
        # Créer un subplot pour chaque colonne
        plt.subplot(num_columns, 1, i)  # Créer les subplots verticalement
        
        # Tracer la série originale et la série débruitée sur le même subplot
        plt.plot(initial_df.index, initial_df[column], label='Original', color='blue')
        plt.plot(denoised_df.index, denoised_df[column], label='Denoised', color='green')
        
        # Configurer le titre et la légende
        plt.title(f'Comparison of Original and Denoised for {column}')
        plt.legend()

    plt.tight_layout()  # Ajuster automatiquement le layout
    plt.show()


