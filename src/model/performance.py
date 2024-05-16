import numpy as np
from src.model.kalman import KalmanModel, s_func

def calculate_rmse(predicted, actual):
    """
    Calculate the root mean square error (RMSE) between the predicted and actual values.

    Parameters:
    predicted (array-like): Predicted values.
    actual (array-like): Actual values.

    Returns:
    float: The RMSE value.
    """
    return np.sqrt(np.mean((predicted - actual) ** 2))

def calculate_future_price(x, s_t, T_minus_t, mu, lambdas, sigmas, kappas, rhos, n_factors, times):
    """
    Calculate the logarithm of the future price based on the number of factors and the model parameters.

    Parameters:
    x (array-like): State variables.
    s_t (float): Current spot price.
    T_minus_t (float): Time to maturity.
    mu (float): Drift term.
    lambdas (array-like): Lambda parameters.
    sigmas (array-like): Sigma parameters.
    kappas (array-like): Kappa parameters (length n_factors - 1).
    rhos (array-like): Correlation matrix.
    n_factors (int): Number of factors.
    times (array-like): Time points.

    Returns:
    float: Logarithm of the future price.
    """
    assert len(lambdas) == n_factors, "Length of lambdas must be equal to n_factors"
    assert len(sigmas) == n_factors, "Length of sigmas must be equal to n_factors"
    assert len(kappas) == n_factors - 1, "Length of kappas must be equal to n_factors - 1"

    if n_factors == 1:
        log_F = s_t + x[0] + mu * times + (mu - lambdas[0] + 0.5 * sigmas[0] ** 2) * T_minus_t
    elif n_factors == 2:
        kappa2 = kappas[0]
        if kappa2 == 0:
            raise ValueError("kappa2 cannot be zero for 2-factor model")
        sum_exp_terms = np.exp(-kappa2 * T_minus_t) * x[1]
        sum_lambda_terms = (lambdas[1] * (1 - np.exp(-kappa2 * T_minus_t))) / kappa2
        sum_sigma_rho_terms = (sigmas[0] * sigmas[1] * rhos[0, 1] * (1 - np.exp(-kappa2 * T_minus_t))) / kappa2
        log_F = (s_t + x[0] + sum_exp_terms + mu * times + (mu - lambdas[0] + 0.5 * sigmas[0] ** 2) * T_minus_t 
                 - sum_lambda_terms + 0.5 * sum_sigma_rho_terms)
    elif n_factors == 3:
        kappa2, kappa3 = kappas
        if kappa2 == 0 or kappa3 == 0:
            raise ValueError("kappa2 and kappa3 cannot be zero for 3-factor model")
        sum_exp_terms = np.sum([np.exp(-kappa * T_minus_t) * x[i + 1] for i, kappa in enumerate([kappa2, kappa3])])
        sum_lambda_terms = np.sum([(lambdas[i + 1] * (1 - np.exp(-kappa * T_minus_t))) / kappa for i, kappa in enumerate([kappa2, kappa3])])
        sum_sigma_rho_terms = 0
        for i in range(2):
            for j in range(i + 1, 3):
                kappa_sum = kappas[i] + kappas[j]
                if kappa_sum == 0:
                    raise ValueError("kappa_sum cannot be zero for 3-factor model")
                sum_sigma_rho_terms += ((sigmas[i + 1] * sigmas[j + 1] * rhos[i, j] * (1 - np.exp(-kappa_sum * T_minus_t))) 
                                        / kappa_sum)
        log_F = (s_t + x[0] + sum_exp_terms + mu * times + (mu - lambdas[0] + 0.5 * sigmas[0] ** 2) * T_minus_t 
                 - sum_lambda_terms + 0.5 * sum_sigma_rho_terms)
    elif n_factors == 4:
        kappa2, kappa3, kappa4 = kappas
        if kappa2 == 0 or kappa3 == 0 or kappa4 == 0:
            raise ValueError("kappa2, kappa3, and kappa4 cannot be zero for 4-factor model")
        sum_exp_terms = np.sum([np.exp(-kappa * T_minus_t) * x[i + 1] for i, kappa in enumerate([kappa2, kappa3, kappa4])])
        sum_lambda_terms = np.sum([(lambdas[i + 1] * (1 - np.exp(-kappa * T_minus_t))) / kappa for i, kappa in enumerate([kappa2, kappa3, kappa4])])
        sum_sigma_rho_terms = 0
        for i in range(3):
            for j in range(i + 1, 4):
                kappa_sum = kappas[i] + kappas[j]
                if kappa_sum == 0:
                    raise ValueError("kappa_sum cannot be zero for 4-factor model")
                sum_sigma_rho_terms += ((sigmas[i + 1] * sigmas[j + 1] * rhos[i, j] * (1 - np.exp(-kappa_sum * T_minus_t))) 
                                        / kappa_sum)
        log_F = (s_t + x[0] + sum_exp_terms + mu * times + (mu - lambdas[0] + 0.5 * sigmas[0] ** 2) * T_minus_t 
                 - sum_lambda_terms + 0.5 * sum_sigma_rho_terms)
    else:
        raise ValueError("This function currently supports up to 4 factors only.")
    
    return log_F

def calculate_performance(n_factors, optimized_params, param_keys, observations, times, maturities):
    """
    Calculate the RMSE for different maturities based on the optimized parameters.

    Parameters:
    n_factors (int): Number of factors.
    optimized_params (array-like): Optimized parameters.
    param_keys (list): List of parameter keys.
    observations (array-like): Observed values.
    times (array-like): Time points.
    maturities (array-like): Maturities.

    Returns:
    list: RMSE values for each maturity.
    """
    model_params = {key: optimized_params[i] for i, key in enumerate(param_keys)}
    model_params['maturities'] = maturities
    model_params['current_time'] = times

    kf_model = KalmanModel(n_factors=n_factors, params=model_params)

    # Construct the correlation matrix
    rho_matrix = np.eye(n_factors)
    rho_keys = [f'rho{i+1}{j+1}' for i in range(1, n_factors) for j in range(i+1, n_factors+1)]
    rho_values = [model_params[key] for key in rho_keys if key in model_params]

    k = 0
    for i in range(1, n_factors):
        for j in range(i + 1, n_factors):
            rho_matrix[i, j] = rho_matrix[j, i] = rho_values[k]
            k += 1

    # Generate kappas with appropriate length
    kappas = [model_params.get(f'kappa{j+1}', 1) for j in range(1, n_factors)]

    rmse_results = []
    for maturity_idx in range(observations.shape[1]):
        predicted_prices = []
        actual_prices = []
        for i in range(len(times)):
            t = times[i]
            s_t = s_func(t)
            x = np.zeros((n_factors, 1))
            x[0] = model_params['x1_initial']
            predicted_price = calculate_future_price(
                x, s_t, maturities[i, maturity_idx], model_params['mu'],
                [model_params.get(f'lambda{j+1}', 0) for j in range(n_factors)],
                [model_params.get(f'sigma{j+1}', 0) for j in range(n_factors)],
                kappas,
                rho_matrix,
                n_factors,
                t
            )
            predicted_prices.append(predicted_price)
            actual_prices.append(observations[i, maturity_idx])

        predicted_prices = np.array(predicted_prices).flatten()
        actual_prices = np.array(actual_prices).flatten()
        
        rmse = calculate_rmse(predicted_prices, actual_prices)
        print(f"RMSE for {n_factors} factors, Maturity {maturity_idx + 1}: {rmse:.2f}%")
        rmse_results.append(rmse)

    return rmse_results
