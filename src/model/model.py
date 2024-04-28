import numpy as np
from src.utility.constant import DELTA

def get_state_transition_matrix(num_factors, K2, K3, K4):
    if num_factors == 1:
        A = np.array([[1]]) 
    elif num_factors == 2:
        A = np.array([[1, 0],
                      [0, np.exp(-K2*DELTA)]])
    elif num_factors == 3:
        A = np.array([[1, 0, 0],
                      [0, np.exp(-K2*DELTA), 0],
                      [0, 0, np.exp(-K3*DELTA)]])
    elif num_factors == 4:
        A = np.array([[1, 0, 0, 0],
                      [0, np.exp(-K2*DELTA), 0, 0],
                      [0, 0, np.exp(-K3*DELTA), 0],
                      [0, 0, 0, np.exp(-K4*DELTA)]])
    else:
        raise ValueError("Unsupported number of factors")
    return A

def get_state_intercept(num_factors, mu, sigma):
    if num_factors == 1:
        a = np.array([mu - 0.5*sigma**2])
    else:
        a = np.zeros(num_factors)
        a[0] = mu - 0.5*sigma**2
    return a

def get_measurement_matrix(num_factors, T, t, K2, K3, K4):
    C = np.zeros((5, num_factors))
    C[:, 0] = 1
    if num_factors > 1:
        for i in range(1, 5):
            C[i, 1] = np.exp(-K2*(T[i]-t))
        if num_factors > 2:
            for i in range(1, 5):
                C[i, 2] = np.exp(-K3*(T[i]-t))
    return C

def get_process_noise_covariance(num_factors, kappa2, kappa3, kappa4, rho12, rho13, rho23, rho14, K2, K3, K4, sigma1, sigma2, sigma3, sigma4):
    """ Génère la matrice de covariance du bruit de processus Ω pour un nombre donné de facteurs. """
    if num_factors == 1:
        Omega = np.array([[DELTA * sigma1**2]])
    elif num_factors == 2:
        Omega = np.array([
            [DELTA * sigma1**2, rho12 * sigma1 * sigma2 * (1 - np.exp(-K2 * DELTA)) / kappa2],
            [rho12 * sigma1 * sigma2 * (1 - np.exp(-K2 * DELTA)) / kappa2, (sigma2**2 / (2 * kappa2)) * (1 - np.exp(-2 * K2 * DELTA))]
        ])
    elif num_factors == 3:
        Omega = np.array([
            [DELTA * sigma1**2, rho12 * sigma1 * sigma2 * (1 - np.exp(-K2 * DELTA)) / kappa2, rho13 * sigma1 * sigma3 * (1 - np.exp(-K3 * DELTA)) / kappa3],
            [rho12 * sigma1 * sigma2 * (1 - np.exp(-K2 * DELTA)) / kappa2, (sigma2**2 / (2 * kappa2)) * (1 - np.exp(-2 * K2 * DELTA)), rho23 * sigma2 * sigma3 * (1 - np.exp(-K3 * DELTA)) / kappa3],
            [rho13 * sigma1 * sigma3 * (1 - np.exp(-K3 * DELTA)) / kappa3, rho23 * sigma2 * sigma3 * (1 - np.exp(-K3 * DELTA)) / kappa3, (sigma3**2 / (2 * kappa3)) * (1 - np.exp(-2 * K3 * DELTA))]
        ])
    elif num_factors == 4:
        Omega = np.array([
            [DELTA * sigma1**2, rho12 * sigma1 * sigma2 * (1 - np.exp(-K2 * DELTA)) / kappa2, rho13 * sigma1 * sigma3 * (1 - np.exp(-K3 * DELTA)) / kappa3, rho14 * sigma1 * sigma4 * (1 - np.exp(-K4 * DELTA)) / kappa4],
            [rho12 * sigma1 * sigma2 * (1 - np.exp(-K2 * DELTA)) / kappa2, (sigma2**2 / (2 * kappa2)) * (1 - np.exp(-2 * K2 * DELTA)), rho23 * sigma2 * sigma3 * (1 - np.exp(-K3 * DELTA)) / kappa3, 0],
            [rho13 * sigma1 * sigma3 * (1 - np.exp(-K3 * DELTA)) / kappa3, rho23 * sigma2 * sigma3 * (1 - np.exp(-K3 * DELTA)) / kappa3, (sigma3**2 / (2 * kappa3)) * (1 - np.exp(-2 * K3 * DELTA)), 0],
            [rho14 * sigma1 * sigma4 * (1 - np.exp(-K4 * DELTA)) / kappa4, 0, 0, (sigma4**2 / (2 * kappa4)) * (1 - np.exp(-2 * K4 * DELTA))]
        ])
    else:
        raise ValueError("Unsupported number of factors for this model")
    return Omega

def get_measurement_constant(num_factors, t, mu, sigma, lambdas, T, s_func):
    """
    Génère le vecteur des constantes ct pour chaque mesure en fonction du nombre de facteurs.
    """
    ct = np.zeros(len(T))
    for i in range(len(T)):
        lambda_value = lambdas.get(i + 1, 0)  
        ct[i] = s_func(t) + (mu + lambda_value - 0.5 * sigma**2) * (T[i] - t)

    return ct

def s_func(t, gamma_cos, gamma_sin, K=2):
    """
    Définit une fonction cyclique déterministe basée sur une série de Fourier tronquée,
    utilisée pour capturer la variation saisonnière.
    """
    result = 0
    for k in range(1, K+1):
        result += gamma_cos[k] * np.cos(2 * np.pi * k * t) + gamma_sin[k] * np.sin(2 * np.pi * k * t)
    return result
