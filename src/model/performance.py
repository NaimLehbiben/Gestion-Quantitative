import numpy as np
from src.model.kalman import KalmanModel, s_func

def calculate_future_price(x, s_t, maturities, mu, lambdas, sigmas, kappas, rhos, n_factors, times):
    future_prices = []

    for T_minus_t in maturities:
        if n_factors > 1:
            sum_exp_terms = np.sum([np.exp(-kappas[i] * T_minus_t) * x[i + 1] for i in range(n_factors - 1)])
            sum_lambda_terms = np.sum([(1 - np.exp(-kappas[i] * T_minus_t)) / kappas[i] * lambdas[i + 1] for i in range(n_factors - 1)])
            sum_sigma_rho_terms = 0
            for i in range(n_factors - 1):
                for j in range(i + 1, n_factors - 1):
                    sum_sigma_rho_terms += ((1 - np.exp(-(kappas[i] + kappas[j]) * T_minus_t)) / (kappas[i] + kappas[j]) * 
                                            sigmas[i + 1] * sigmas[j + 1] * rhos[i, j])
        else:
            sum_exp_terms = 0
            sum_lambda_terms = 0
            sum_sigma_rho_terms = 0

        log_F = (s_t + x[0] + sum_exp_terms + mu * times + 
                (mu - lambdas[0] + 0.5 * sigmas[0]**2) * T_minus_t - 
                sum_lambda_terms + 0.5 * sum_sigma_rho_terms)

        future_prices.append(log_F)

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

    kappas = [model_params.get(f'kappa{j+1}', 0) for j in range(1, n_factors)] if n_factors > 1 else []

    predicted_prices = []
    actual_prices = []
    for i in range(len(times)):
        t = times[i]
        s_t = s_func(t)
        x = np.zeros((n_factors, 1))
        x[0] = model_params['x1_initial']
        predicted_price = calculate_future_price(
            x, s_t, maturities[i], model_params['mu'],
            [model_params[f'lambda{j+1}'] for j in range(n_factors)],
            [model_params[f'sigma{j+1}'] for j in range(n_factors)],
            kappas,
            rho_matrix,
            n_factors,
            t
        )
        predicted_prices.append(predicted_price)
        actual_prices.append(observations[i, :])

    predicted_prices = np.array(predicted_prices)
    actual_prices = np.array(actual_prices)
    
    for maturity_idx in range(predicted_prices.shape[1]):
        rmse_percentage = calculate_rmse(predicted_prices[:, maturity_idx], actual_prices[:, maturity_idx])
        print(f"RMSE for {n_factors} factors, Maturity {maturity_idx + 1}: {rmse_percentage:.2f}%")


