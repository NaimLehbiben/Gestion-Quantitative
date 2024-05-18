from scipy.optimize import minimize
from src.model.kalman import KalmanModel
from src.utility.parameter import calculate_num_parameters
import numpy as np
import scipy.stats as stats

def objective(params, observations, times, maturities, n_factors, seasonal_coeffs):
    param_keys = ['mu', 'sigma1', 'lambda1', 'kappa2', 'sigma2', 'lambda2', 'rho12',
                  'kappa3', 'sigma3', 'lambda3', 'rho13', 'rho23',
                  'kappa4', 'sigma4', 'lambda4', 'rho14', 'rho24', 'rho34']
    num_params = calculate_num_parameters(n_factors)
    param_keys = param_keys[:num_params]

    model_params = {key: params[i] for i, key in enumerate(param_keys)}
    model_params['maturities'] = maturities
    model_params['current_time'] = times
    model_params['seasonal_coeffs'] = seasonal_coeffs

    for key in model_params:
        if 'kappa' in key and model_params[key] == 0 and n_factors > 1:
            raise ValueError(f"{key} cannot be zero for {n_factors}-factor model")

    model = KalmanModel(n_factors=n_factors, params=model_params, seasonal_coeffs=seasonal_coeffs)
    return model.compute_likelihood(observations, times, maturities)

def optimize_model(observations, times, maturities, n_factors, initial_guess, seasonal_coeffs, reg_lambda=1e-6):
    initial_result = minimize(
        objective,
        initial_guess,
        args=(observations, times, maturities, n_factors, seasonal_coeffs),
        method='Nelder-Mead',
        options={'maxiter': 200}
    )

    final_result = minimize(
        objective,
        initial_result.x,
        args=(observations, times, maturities, n_factors, seasonal_coeffs),
        method='BFGS',
        options={'maxiter': 10}
    )

    hessian_inv = final_result.hess_inv
    if isinstance(hessian_inv, np.ndarray):
        covariance_matrix = hessian_inv
    else:
        covariance_matrix = hessian_inv.todense().astype(np.float64)
    
    # Ajouter une petite valeur positive à la diagonale
    covariance_matrix += np.eye(covariance_matrix.shape[0]) * reg_lambda

    # Assurer la positivité semi-définie de la matrice de covariance
    try:
        # Utiliser la décomposition de Cholesky pour vérifier la positivité semi-définie
        np.linalg.cholesky(covariance_matrix)
    except np.linalg.LinAlgError:
        # Si la matrice n'est pas positive semi-définie, ajouter une régularisation supplémentaire
        eigvals = np.linalg.eigvals(covariance_matrix)
        min_eigval = min(eigvals)
        if min_eigval < 0:
            covariance_matrix += np.eye(covariance_matrix.shape[0]) * (-min_eigval + reg_lambda)

    try:
        np.linalg.cholesky(covariance_matrix)
        std_errors = np.sqrt(np.diag(covariance_matrix))
    except np.linalg.LinAlgError:
        std_errors = np.full(covariance_matrix.shape[0], np.inf)

    z_values = final_result.x / std_errors
    p_values = [2 * (1 - stats.norm.cdf(np.abs(z))) for z in z_values]


    return final_result
