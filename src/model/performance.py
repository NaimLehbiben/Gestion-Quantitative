import numpy as np
import scipy.stats as stats
from src.model.kalman import KalmanModel, s_func

def calculate_future_price(x, s_t, maturities, mu, lambdas, sigmas, kappas, rhos):
    N = len(x)
    future_prices = []

    for T_minus_t in maturities:
        sum_exp_terms = np.sum([np.exp(-kappas[i] * T_minus_t) * x[i] for i in range(1, N)])
        sum_lambda_terms = np.sum([(1 - np.exp(-kappas[i] * T_minus_t)) / kappas[i] * lambdas[i] for i in range(1, N)])
        sum_sigma_rho_terms = 0
        for i in range(1, N):
            for j in range(i + 1, N):
                sum_sigma_rho_terms += ((1 - np.exp(-(kappas[i] + kappas[j]) * T_minus_t)) / (kappas[i] + kappas[j]) * 
                                        sigmas[i] * sigmas[j] * rhos[i, j])

        log_F = (s_t + x[0] + sum_exp_terms + mu * T_minus_t + 
                (mu - lambdas[0] + 0.5 * sigmas[0]**2) * T_minus_t - 
                sum_lambda_terms + 0.5 * sum_sigma_rho_terms)

        future_prices.append(np.exp(log_F))

    return np.array(future_prices)

def calculate_rmse(predicted, actual):
    return np.sqrt(np.mean((predicted - actual)**2))

def calculate_performance(n_factors, optimized_params, param_keys, observations, times, maturities):
    model_params = {key: optimized_params[i] for i, key in enumerate(param_keys)}
    model_params['maturities'] = maturities
    model_params['current_time'] = times

    kf_model = KalmanModel(n_factors=n_factors, params=model_params)

    rho_matrix = np.eye(n_factors)
    rho_keys = [f'rho{i+1}{j+1}' for i in range(1, n_factors) for j in range(i+1, n_factors+1)]
    rho_values = [model_params[key] for key in rho_keys if key in model_params]

    k = 0
    for i in range(1, n_factors):
        for j in range(i+1, n_factors):
            rho_matrix[i, j] = rho_matrix[j, i] = rho_values[k]
            k += 1

    predicted_prices = []
    actual_prices = []
    for i in range(len(times)):
        t = times[i]
        s_t = s_func(t)
        x = kf_model.kf.x[:, 0]
        predicted_price = calculate_future_price(
            x, s_t, maturities[i], model_params['mu'],
            [model_params[f'lambda{j+1}'] for j in range(n_factors)],
            [model_params[f'sigma{j+1}'] for j in range(n_factors)],
            [model_params.get(f'kappa{j+1}', 0) for j in range(1, n_factors)],
            rho_matrix
        )
        predicted_prices.append(predicted_price)
        actual_prices.append(observations[i, :])

    predicted_prices = np.array(predicted_prices)
    actual_prices = np.array(actual_prices)
    rmse = calculate_rmse(predicted_prices, actual_prices)
    print(f"RMSE for {n_factors} factors: {rmse}")

    std_errors = np.sqrt(np.diag(kf_model.kf.P))
    z_values = optimized_params[:len(param_keys)] / std_errors[:len(param_keys)]
    p_values = [2 * (1 - stats.norm.cdf(np.abs(z))) for z in z_values]

    for i, (param, std_err, p_value) in enumerate(zip(optimized_params[:len(param_keys)], std_errors[:len(param_keys)], p_values[:len(param_keys)])):
        print(f"Parameter {param_keys[i]}: estimate={param}, std_error={std_err}, p_value={p_value}")
