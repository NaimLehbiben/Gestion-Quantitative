import numpy as np
from filterpy.kalman import KalmanFilter
from src.utility.parameter import DELTA


def s_func(t):
    return -0.0088 * np.cos(2 * np.pi * t) + 0.0035 * np.cos(2 * np.pi * 2 * t) + \
           0.0344 * np.sin(2 * np.pi * t) - 0.0098 * np.sin(2 * np.pi * 2 * t)

class KalmanModel:
    def __init__(self, n_factors, params):
        self.kf = KalmanFilter(dim_x=n_factors, dim_z=5)
        self.n_factors = n_factors
        self.params = params
        self.configure_matrices()

    def configure_matrices(self):
        self.kf.F = self.get_state_transition_matrix()
        self.kf.H = self.get_measurement_matrix()
        self.a = self.get_state_intercept()
        self.kf.x = np.zeros((self.n_factors, 1))
        
        self.kf.P = np.eye(self.n_factors)
        self.kf.P[0, 0] = 1e4  
        for i in range(1, self.n_factors):
            self.kf.P[i, i] = 1.0

        self.kf.Q = self.get_process_noise_covariance()
        self.kf.R = np.eye(5) * 0.01

    def get_state_transition_matrix(self):
        A = np.eye(self.n_factors)
        A[0, 0] = 1  
        for i in range(1, self.n_factors):
            kappa = self.params.get(f'kappa{i+1}', 0)
            A[i, i] = np.exp(-kappa * DELTA)
        return A

    def get_state_intercept(self):
        mu = self.params.get('mu')
        sigma1 = self.params.get('sigma1')
        a = np.zeros((self.n_factors, 1))
        a[0, 0] = mu - 0.5 * sigma1**2
        return a

    def get_measurement_matrix(self):
        T = self.params['maturities']
        C = np.zeros((5, self.n_factors))
        C[:, 0] = 1
        for i in range(1, self.n_factors):
            kappa = self.params.get(f'kappa{i+1}', 0)
            decay_factors = np.exp(-kappa * T)
            C[:, i] = decay_factors
        return C

    def get_process_noise_covariance(self):
        Q = np.zeros((self.n_factors, self.n_factors))
        for i in range(self.n_factors):
            sigma_i = self.params.get(f'sigma{i+1}', 0)
            kappa_i = self.params.get(f'kappa{i+1}', 0)
            if i == 0:
                Q[i, i] = sigma_i**2 * DELTA  
            else:
                Q[i, i] = (sigma_i**2 * (1 - np.exp(-2 * kappa_i * DELTA))) / (2 * kappa_i)
            for j in range(i + 1, self.n_factors):
                sigma_j = self.params.get(f'sigma{j+1}', 0)
                kappa_j = self.params.get(f'kappa{j+1}', 0)
                rho_ij = self.params.get(f'rho{i+1}{j+1}', 0)
                term = (rho_ij * sigma_i * sigma_j * (1 - np.exp(-(kappa_i + kappa_j) * DELTA))) / (kappa_i + kappa_j)
                Q[i, j] = Q[j, i] = term
        return Q

    def compute_ct(self, s_t, maturities):
        mu = self.params.get('mu', 0)
        terms = []
        for i in range(len(maturities)):
            factor_index = i % 4 + 1
            lambda_i = self.params.get(f'lambda{factor_index}', 0)
            sigma_i = self.params.get(f'sigma{factor_index}', 0)
            term = s_t + (mu + lambda_i - 0.5 * sigma_i**2) * maturities[i]
            terms.append(term)
        c_t = np.array(terms).reshape(-1, 1)
        return c_t

    def compute_likelihood(self, observations, times, maturities, exclude_first_n=0.01):
        start_index = int(exclude_first_n * len(observations))
        total_log_likelihood = 0.0
        for i in range(start_index, len(observations)):
            t = times[i]
            s_t = s_func(t)
            c_t = self.compute_ct(s_t, maturities[i])
            z = observations[i].reshape(-1, 1)
            self.kf.predict(u=self.a)
            self.kf.update(z - c_t)
            total_log_likelihood += self.kf.log_likelihood
        return -total_log_likelihood