from scipy.optimize import minimize
from src.model.kalman import KalmanModel
from src.utility.parameter import calculate_num_parameters

def objective(params, observations, times, maturities, n_factors, seasonal_coeffs):
    param_keys = ['x1_initial', 'mu', 'sigma1', 'lambda1', 'kappa2', 'sigma2', 'lambda2', 'rho12',
                  'kappa3', 'sigma3', 'lambda3', 'rho13', 'rho23',
                  'kappa4', 'sigma4', 'lambda4', 'rho14', 'rho24', 'rho34']
    num_params = calculate_num_parameters(n_factors) + 1
    model_params = {key: params[i] for i, key in enumerate(param_keys[:num_params])}
    model_params['maturities'] = maturities
    model_params['current_time'] = times
    model_params['seasonal_coeffs'] = seasonal_coeffs

    for key in model_params:
        if 'kappa' in key and model_params[key] == 0 and n_factors > 1:
            raise ValueError(f"{key} cannot be zero for {n_factors}-factor model")

    model = KalmanModel(n_factors=n_factors, params=model_params, seasonal_coeffs=seasonal_coeffs)
    return model.compute_likelihood(observations, times, maturities)

def optimize_model(observations, times, maturities, n_factors, initial_guess, seasonal_coeffs):
    initial_result = minimize(
        objective,
        initial_guess,
        args=(observations, times, maturities, n_factors, seasonal_coeffs),
        method='Nelder-Mead',
        options={'maxiter': 1}
    )

    final_result = minimize(
        objective,
        initial_result.x,
        args=(observations, times, maturities, n_factors, seasonal_coeffs),
        method='BFGS',
        options={'maxiter': 1}
    )
    
    return final_result
