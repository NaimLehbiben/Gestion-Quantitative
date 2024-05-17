import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pywt


def denoise_signal(signal, wavelet='db1',level =1):

    coeffs = pywt.wavedec(signal, wavelet)
    
    def madev(d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    sigma = madev(coeffs[-level])/ 0.6745
    
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    
    
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)

    return denoised_signal

def denoise_all_signal(df_price, wavelet='db1',level =1):


    denoised_price = {}
    for column in df_price.columns:
        signal = df_price[column].values
        denoised_signal = denoise_signal(signal, wavelet=wavelet,level = level)  
        denoised_price[column] = denoised_signal[:len(df_price)]  

    
    denoised_price_df = pd.DataFrame(denoised_price, index=df_price.index)

    return denoised_price_df