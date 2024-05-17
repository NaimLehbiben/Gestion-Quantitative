import numpy as np
import pywt

def apply_wavelet_decomposition(data, wavelet='db2', mode='symmetric'):
    """
    Effectue la décomposition par ondelettes du signal.

    Parameters:
    data (array-like): Série de données à décomposer.
    wavelet (str): Nom de l'ondelette à utiliser.
    mode (str): Mode de bordure à utiliser dans la décomposition.

    Returns:
    list: Coefficients de la décomposition par ondelettes.
    """
    coeffs = pywt.wavedec(data, wavelet, mode=mode)
    return coeffs

def calculate_universal_threshold(coeffs, data_length):
    """
    Calcule le seuil universel basé sur l'écart absolu médian (MAD).

    Parameters:
    coeffs (list): Coefficients de la décomposition par ondelettes.
    data_length (int): Longueur de la série de données.

    Returns:
    float: Seuil calculé.
    """
    sigma_mad = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = np.sqrt(2 * sigma_mad**2 * np.log(data_length))
    return threshold

def custom_soft_threshold(w, delta):
    """
    Applique un seuil doux personnalisé aux coefficients.

    Parameters:
    w (array-like): Coefficients à seuiler.
    delta (float): Valeur du seuil.

    Returns:
    array-like: Coefficients seuilés.
    """
    return np.sign(w) * 0.5 * (np.abs(w) - delta + np.abs(np.abs(w) - delta))

def threshold_coeffs(coeffs, threshold):
    """
    Applique le seuil aux coefficients détaillés.

    Parameters:
    coeffs (list): Coefficients de la décomposition par ondelettes.
    threshold (float): Valeur du seuil.

    Returns:
    list: Coefficients seuilés.
    """
    thresholded_coeffs = [coeffs[0]] + [custom_soft_threshold(detail_coeff, threshold) for detail_coeff in coeffs[1:]]
    return thresholded_coeffs

def reconstruct_signal_from_coeffs(thresholded_coeffs, wavelet='db2', mode='symmetric'):
    """
    Reconstruit le signal à partir des coefficients seuilés.

    Parameters:
    thresholded_coeffs (list): Coefficients seuilés.
    wavelet (str): Nom de l'ondelette à utiliser.
    mode (str): Mode de bordure à utiliser dans la reconstruction.

    Returns:
    array-like: Signal reconstruit.
    """
    reconstructed_signal = pywt.waverec(thresholded_coeffs, wavelet, mode=mode)
    return reconstructed_signal

