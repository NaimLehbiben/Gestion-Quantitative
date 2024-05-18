import numpy as np
from filterpy.kalman import KalmanFilter
from src.utility.parameter import DELTA

def s_func(t, coeff_Cos1, coeff_Sin1, coeff_Cos2, coeff_Sin2):
    return coeff_Cos1 * np.cos(2 * np.pi * t) + coeff_Sin1 * np.sin(2 * np.pi * t) + \
           coeff_Cos2 * np.cos(2 * np.pi * 2 * t) + coeff_Sin2 * np.sin(2 * np.pi * 2 * t)

class KalmanModel:
    def __init__(self, n_factors, params, seasonal_coeffs):
        self.kf = KalmanFilter(dim_x=n_factors, dim_z=5)
        self.n_factors = n_factors
        self.params = params
        self.seasonal_coeffs = seasonal_coeffs
        self.configure_matrices()

    def configure_matrices(self):
        self.kf.F = self.get_state_transition_matrix()
        self.kf.H = self.get_measurement_matrix()
        self.a = self.get_state_intercept()
        self.kf.x = np.zeros((self.n_factors, 1))

        self.kf.P = np.eye(self.n_factors)
        self.kf.P[0, 0] = 1e4  # Grande valeur pour la variable non stationnaire
        for i in range(1, self.n_factors):
            self.kf.P[i, i] = 1.0  # Valeur pour les variables stationnaires
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
            for j in range(5):
                decay_factors = np.exp(-kappa * T[:, j])
                C[j, i] = decay_factors[j]
        return C

    def get_process_noise_covariance(self):
        n_factors = self.n_factors
        params = self.params

        Q = np.zeros((n_factors, n_factors))
        
        if n_factors == 1:
            sigma1 = params.get('sigma1', 0)
            Q[0, 0] = sigma1**2 * DELTA
        elif n_factors == 2:
            sigma1 = params.get('sigma1', 0)
            sigma2 = params.get('sigma2', 0)
            kappa2 = params.get('kappa2', 0)
            rho12 = params.get('rho12', 0)

            Q[0, 0] = sigma1**2 * DELTA
            Q[1, 1] = (sigma2**2 * (1 - np.exp(-2 * kappa2 * DELTA))) / (2 * kappa2)
            Q[0, 1] = Q[1, 0] = (rho12 * sigma1 * sigma2 * (1 - np.exp(-(kappa2 + kappa2) * DELTA))) / (kappa2 + kappa2)
        elif n_factors == 3:
            sigma1 = params.get('sigma1', 0)
            sigma2 = params.get('sigma2', 0)
            sigma3 = params.get('sigma3', 0)
            kappa2 = params.get('kappa2', 0)
            kappa3 = params.get('kappa3', 0)
            rho12 = params.get('rho12', 0)
            rho13 = params.get('rho13', 0)
            rho23 = params.get('rho23', 0)

            Q[0, 0] = sigma1**2 * DELTA
            Q[1, 1] = (sigma2**2 * (1 - np.exp(-2 * kappa2 * DELTA))) / (2 * kappa2)
            Q[2, 2] = (sigma3**2 * (1 - np.exp(-2 * kappa3 * DELTA))) / (2 * kappa3)
            Q[0, 1] = Q[1, 0] = (rho12 * sigma1 * sigma2 * (1 - np.exp(-(kappa2 + kappa2) * DELTA))) / (kappa2 + kappa2)
            Q[0, 2] = Q[2, 0] = (rho13 * sigma1 * sigma3 * (1 - np.exp(-(kappa3 + kappa3) * DELTA))) / (kappa3 + kappa3)
            Q[1, 2] = Q[2, 1] = (rho23 * sigma2 * sigma3 * (1 - np.exp(-(kappa2 + kappa3) * DELTA))) / (kappa2 + kappa3)
        elif n_factors == 4:
            sigma1 = params.get('sigma1', 0)
            sigma2 = params.get('sigma2', 0)
            sigma3 = params.get('sigma3', 0)
            sigma4 = params.get('sigma4', 0)
            kappa2 = params.get('kappa2', 0)
            kappa3 = params.get('kappa3', 0)
            kappa4 = params.get('kappa4', 0)
            rho12 = params.get('rho12', 0)
            rho13 = params.get('rho13', 0)
            rho14 = params.get('rho14', 0)
            rho23 = params.get('rho23', 0)
            rho24 = params.get('rho24', 0)
            rho34 = params.get('rho34', 0)

            Q[0, 0] = sigma1**2 * DELTA
            Q[1, 1] = (sigma2**2 * (1 - np.exp(-2 * kappa2 * DELTA))) / (2 * kappa2)
            Q[2, 2] = (sigma3**2 * (1 - np.exp(-2 * kappa3 * DELTA))) / (2 * kappa3)
            Q[3, 3] = (sigma4**2 * (1 - np.exp(-2 * kappa4 * DELTA))) / (2 * kappa4)
            Q[0, 1] = Q[1, 0] = (rho12 * sigma1 * sigma2 * (1 - np.exp(-(kappa2 + kappa2) * DELTA))) / (kappa2 + kappa2)
            Q[0, 2] = Q[2, 0] = (rho13 * sigma1 * sigma3 * (1 - np.exp(-(kappa3 + kappa3) * DELTA))) / (kappa3 + kappa3)
            Q[0, 3] = Q[3, 0] = (rho14 * sigma1 * sigma4 * (1 - np.exp(-(kappa4 + kappa4) * DELTA))) / (kappa4 + kappa4)
            Q[1, 2] = Q[2, 1] = (rho23 * sigma2 * sigma3 * (1 - np.exp(-(kappa2 + kappa3) * DELTA))) / (kappa2 + kappa3)
            Q[1, 3] = Q[3, 1] = (rho24 * sigma2 * sigma4 * (1 - np.exp(-(kappa2 + kappa4) * DELTA))) / (kappa2 + kappa4)
            Q[2, 3] = Q[3, 2] = (rho34 * sigma3 * sigma4 * (1 - np.exp(-(kappa3 + kappa4) * DELTA))) / (kappa3 + kappa4)
        
        return Q

    def compute_ct(self, s_t, maturities):
        mu = self.params.get('mu', 0)
        sigma1 = self.params.get('sigma1', 0)

        terms = []
        for i in range(len(maturities)):
            factor_index = i % 4 + 1
            lambda_i = self.params.get(f'lambda{factor_index}', 0)
            term = s_t + (mu + lambda_i - 0.5 * sigma1**2) * maturities[i]
            terms.append(term)

        c_t = np.array(terms).reshape(-1, 1)
        return c_t

    def compute_likelihood(self, observations, times, maturities, exclude_first_n=0.01):
        start_index = int(exclude_first_n * len(observations))
        total_log_likelihood = 0.0

        if observations.shape[1] != 5:
            raise ValueError(f"Each observation should have 5 elements, but got shape {observations.shape}")

        for i in range(start_index, len(observations)):
            t = times[i]
            s_t = s_func(t, self.seasonal_coeffs['coeff_Cos1'],
                         self.seasonal_coeffs['coeff_Sin1'], self.seasonal_coeffs['coeff_Cos2'],
                         self.seasonal_coeffs['coeff_Sin2'])
            c_t = self.compute_ct(s_t, maturities[i])
            z = observations[i].reshape(-1, 1)
            self.kf.predict(u=np.array([self.a]))
            self.kf.update(z - c_t)
            total_log_likelihood += self.kf.log_likelihood

        return -total_log_likelihood
