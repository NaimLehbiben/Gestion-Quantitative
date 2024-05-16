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
        self.kf.P[0, 0] = 1e4  # Grand valeur pour la variable non stationnaire
        for i in range(1, self.n_factors):
            self.kf.P[i, i] = 1.0  # Valeur en état stationnaire pour les variables stationnaires

        self.kf.Q = self.get_process_noise_covariance()
        self.kf.R = np.eye(5) * 0.01

    def get_state_transition_matrix(self):
        """
        Generates the state transition matrix for the Kalman Filter model.

        Returns:
        np.ndarray: The state transition matrix.
        """
        A = np.eye(self.n_factors)
        
        
        if A.shape != (self.n_factors, self.n_factors):
            raise ValueError(f"A should have shape ({self.n_factors}, {self.n_factors}), but got shape {A.shape}")

        A[0, 0] = 1
        for i in range(1, self.n_factors):
            kappa = self.params.get(f'kappa{i+1}', 0)

            if not isinstance(kappa, (int, float)):
                raise ValueError(f"kappa{i+1} must be a number, but got {type(kappa)}")

            A[i, i] = np.exp(-kappa * DELTA)

        return A


    def get_state_intercept(self):
        """
        Generates the state intercept vector for the initial state.

        Returns:
        np.ndarray: The state intercept vector.
        """
        mu = self.params.get('mu')
        sigma1 = self.params.get('sigma1')

        
        if not isinstance(mu, (int, float)):
            raise ValueError(f"mu must be a number, but got {type(mu)}")
        if not isinstance(sigma1, (int, float)):
            raise ValueError(f"sigma1 must be a number, but got {type(sigma1)}")

        a = np.zeros((self.n_factors, 1))
        a[0, 0] = mu - 0.5 * sigma1**2

        
        if a.shape != (self.n_factors, 1):
            raise ValueError(f"a should have shape ({self.n_factors}, 1), but got shape {a.shape}")

        return a

    def get_measurement_matrix(self):
        """
        Generates the measurement matrix used in the Kalman Filter.

        Returns:
        np.ndarray: The measurement matrix.
        """
        T = self.params['maturities']
        
        
        if T.shape[1] != 5:
            raise ValueError(f"Maturities should have 5 columns, but got shape {T.shape}")

        C = np.zeros((5, self.n_factors))
        C[:, 0] = 1


        for i in range(1, self.n_factors):
            kappa = self.params.get(f'kappa{i+1}', 0)

            
            if not isinstance(kappa, (int, float)):
                raise ValueError(f"kappa{i+1} must be a number, but got {type(kappa)}")

            for j in range(5):
                decay_factors = np.exp(-kappa * T[:, j])
                C[j, i] = decay_factors[j]

        return C


    def get_process_noise_covariance(self):
        """
        Generates the process noise covariance matrix for the Kalman Filter, with correlation limited to 95%.

        Returns:
            np.ndarray: Process noise covariance matrix.
        """
        n_factors = self.n_factors
        params = self.params

        Q = np.zeros((n_factors, n_factors))
        
        for i in range(n_factors):
            sigma_i = params.get(f'sigma{i+1}', 0)
            kappa_i = params.get(f'kappa{i+1}', 0)
            
            
            if not isinstance(sigma_i, (int, float)):
                raise ValueError(f"sigma{i+1} must be a number, but got {type(sigma_i)}")
            if not isinstance(kappa_i, (int, float)):
                raise ValueError(f"kappa{i+1} must be a number, but got {type(kappa_i)}")

            if i == 0:
                Q[i, i] = sigma_i**2 * DELTA  
            else:
                if kappa_i == 0:
                    raise ValueError(f"kappa for factor {i+1} cannot be zero for mean-reverting processes.")
                Q[i, i] = (sigma_i**2 * (1 - np.exp(-2 * kappa_i * DELTA))) / (2 * kappa_i)
            
            for j in range(i + 1, n_factors):
                sigma_j = params.get(f'sigma{j+1}', 0)
                kappa_j = params.get(f'kappa{j+1}', 0)
                
                
                if not isinstance(sigma_j, (int, float)):
                    raise ValueError(f"sigma{j+1} must be a number, but got {type(sigma_j)}")
                if not isinstance(kappa_j, (int, float)):
                    raise ValueError(f"kappa{j+1} must be a number, but got {type(kappa_j)}")

                if kappa_i + kappa_j == 0:
                    continue  

                rho_ij = params.get(f'rho{i+1}{j+1}', 0)

                
                if not isinstance(rho_ij, (int, float)):
                    raise ValueError(f"rho{i+1}{j+1} must be a number, but got {type(rho_ij)}")


                if kappa_i == 0 or kappa_j == 0:
                    effective_kappa = kappa_j if kappa_i == 0 else kappa_i
                    term = (rho_ij * sigma_i * sigma_j * (1 - np.exp(-effective_kappa * DELTA))) / effective_kappa
                else:
                    term = (rho_ij * sigma_i * sigma_j * (1 - np.exp(-(kappa_i + kappa_j) * DELTA))) / (kappa_i + kappa_j)
                Q[i, j] = Q[j, i] = term

        return Q

    
    def compute_ct(self, s_t, maturities):
        """
        Computes the state intercept term c_t based on the seasonal component and maturities.

        Args:
        s_t (float): The seasonal component at time t.
        maturities (np.ndarray): Array of maturities for the factors.

        Returns:
        np.ndarray: The state intercept term c_t with shape (5, 1).

        Raises:
        ValueError: If the resulting c_t does not have the shape (5, 1).
        """
        mu = self.params.get('mu', 0)

        terms = []
        for i in range(len(maturities)):
            
            factor_index = i % 4 + 1
            lambda_i = self.params.get(f'lambda{factor_index}', 0)
            sigma_i = self.params.get(f'sigma{factor_index}', 0)
            term = s_t + (mu + lambda_i - 0.5 * sigma_i**2) * maturities[i]
            terms.append(term)

        c_t = np.array(terms).reshape(-1, 1)

        if c_t.shape != (5, 1):
            raise ValueError(f"Each intercept should have shape (5, 1), but got shape {c_t.shape}")
        
        return c_t



    def compute_likelihood(self, observations, times, maturities, exclude_first_n=0.01):
        """
        Computes the negative log-likelihood of the observations given the model parameters.

        Args:
        observations (np.ndarray): The observed data with shape (n_samples, 5).
        times (np.ndarray): The time points corresponding to the observations.
        maturities (np.ndarray): The maturities for the factors.
        exclude_first_n (float, optional): Fraction of the initial observations to exclude from likelihood computation.

        Returns:
        float: The negative log-likelihood of the observations.

        Raises:
        ValueError: If observations do not have 5 elements per sample.
        """
        start_index = int(exclude_first_n * len(observations))
        total_log_likelihood = 0.0

        if observations.shape[1] != 5:
            raise ValueError(f"Each observation should have 5 elements, but got shape {observations.shape}")

        for i in range(start_index, len(observations)):
            t = times[i]
            s_t = s_func(t)
            c_t = self.compute_ct(s_t, maturities[i])
            z = observations[i].reshape(-1, 1)
            self.kf.predict(u=np.array([self.a]))
            self.kf.update(z- c_t)
            total_log_likelihood += self.kf.log_likelihood

        return -total_log_likelihood
