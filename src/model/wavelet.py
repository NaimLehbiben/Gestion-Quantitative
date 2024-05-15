import numpy as np
import pywt

def apply_wavelet_decomposition(data, wavelet='db1', mode='symmetric'):
    coeffs = pywt.wavedec(data, wavelet, mode=mode)
    return coeffs

def calculate_mad_based_threshold(coeffs, data_length, s=1):
    sigma_mad = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = np.sqrt(s * sigma_mad * np.log(data_length))
    return threshold

def custom_soft_threshold(w, delta):
    return np.sign(w) * 0.5 * (np.abs(w) - delta + np.abs(np.abs(w) - delta))

def threshold_coeffs(coeffs, threshold):
    thresholded_coeffs = [coeffs[0]] + [custom_soft_threshold(detail_coeff, threshold) for detail_coeff in coeffs[1:]]
    return thresholded_coeffs

def reconstruct_signal_from_coeffs(thresholded_coeffs, wavelet='db1', mode='symmetric'):
    reconstructed_signal = pywt.waverec(thresholded_coeffs, wavelet, mode=mode)
    return reconstructed_signal
