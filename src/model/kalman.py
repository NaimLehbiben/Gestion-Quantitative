import numpy as np
from filterpy.kalman import KalmanFilter
from src.utility.constant import DELTA

def s_func(t):
    """
    Fonction saisonnière pour moduler les signaux périodiques basés sur le temps 't'.

    Args:
    t (float): Le temps en années.

    Returns:
    float: La valeur calculée de la fonction saisonnière à l'instant t.
    """
    return -0.0228 * np.cos(2 * np.pi * t) + 0.0029 * np.cos(2 * np.pi * 2 * t) + \
           0.0081 * np.sin(2 * np.pi * t) + 0.0054 * np.sin(2 * np.pi * 2 * t)

class KalmanModel:
    """
    Un modèle de filtre de Kalman pour estimer les états dynamiques basés sur des observations temporelles.

    Attributes:
    kf (KalmanFilter): L'instance du filtre de Kalman.
    n_factors (int): Le nombre de facteurs dans le modèle.
    params (dict): Dictionnaire des paramètres du modèle.
    """
    def __init__(self, n_factors, params):
        self.kf = KalmanFilter(dim_x=n_factors, dim_z=5)  
        self.n_factors = n_factors
        self.params = params
        self.configure_matrices()

    def configure_matrices(self):
        """
        Configure les matrices utilisées dans le filtre de Kalman en utilisant les paramètres du modèle.
        """
        self.kf.F = self.get_state_transition_matrix()
        self.kf.H = self.get_measurement_matrix()
        self.a = self.get_state_intercept()  
        self.kf.x = np.zeros((self.n_factors, 1))  
        self.kf.P = np.full((self.n_factors, self.n_factors), 1e6)  # Grande covariance initiale recommandée par Durbin et Watson (2006)
        self.kf.Q = self.get_process_noise_covariance()
        self.kf.R = np.eye(5) * 0.01  

    def get_state_transition_matrix(self):
        """
        Génère la matrice de transition d'état du modèle de Kalman.

        Returns:
        np.ndarray: La matrice de transition d'état.
        """
        A = np.eye(self.n_factors)
        for i in range(1, self.n_factors):
            kappa = self.params.get(f'kappa{i+1}', 0)
            A[i, i] = np.exp(-kappa * DELTA)
        return A

    def get_state_intercept(self):
        """
        Génère le vecteur d'interception de l'état initial.

        Returns:
        np.ndarray: Le vecteur d'interception d'état.
        """
        a = np.zeros((self.n_factors, 1))
        a[0, 0] = self.params['mu'] - 0.5 * self.params['sigma1']**2
        return a

    def get_measurement_matrix(self):
        """
        Génère la matrice de mesure utilisée dans le filtre de Kalman.

        Returns:
        np.ndarray: La matrice de mesure.
        """
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
        """
        Génère la matrice de covariance du bruit de processus.

        Returns:
        np.ndarray: La matrice de covariance du bruit de processus.
        """
        Q = np.zeros((self.n_factors, self.n_factors))
        for i in range(self.n_factors):
            sigma_i = self.params.get(f'sigma{i+1}', 0)
            kappa_i = self.params.get(f'kappa{i+1}', 0)
            if kappa_i == 0:
                Q[i, i] = sigma_i**2 * DELTA  
            else:
                Q[i, i] = sigma_i**2 * (1 - np.exp(-2 * kappa_i * DELTA)) / (2 * kappa_i)
            for j in range(i + 1, self.n_factors):
                sigma_j = self.params.get(f'sigma{j+1}', 0)
                kappa_j = self.params.get(f'kappa{j+1}', 0)
                rho_ij = self.params.get(f'rho{i+1}{j+1}', 0)
                if kappa_i + kappa_j == 0:
                    term = 0  
                else:
                    term = (rho_ij * sigma_i * sigma_j * (1 - np.exp(- (kappa_i + kappa_j) * DELTA))) / (kappa_i + kappa_j)
                Q[i, j] = Q[j, i] = term
        return Q

    def compute_likelihood(self, observations, times, maturities):
        """
        Calcule la vraisemblance des observations données en utilisant le filtre de Kalman.

        Args:
        observations (np.ndarray): Les observations.
        times (np.ndarray): Les instants des observations.
        maturities (np.ndarray): Les maturités correspondantes.

        Returns:
        float: La vraisemblance négative des observations.
        """
        l1 = l2 = 0.0
        for i, z in enumerate(observations):
            t = times[i]
            s_t = s_func(t)
            c_t = np.array([s_t + (self.params['mu'] - self.params['lambda1'] + 0.5 * self.params['sigma1'] ** 2) * m for m in maturities[i]])
            c_t = c_t.reshape(-1, 1)  
            z = z.reshape(-1, 1)
            self.kf.predict(u=np.array([self.a]))
            self.kf.update(z - c_t)
            l1 += np.log(np.linalg.det(self.kf.S))
            l2 += (self.kf.y.T @ np.linalg.inv(self.kf.S) @ self.kf.y)
        n_timesteps = len(observations)
        likelihood = -0.5 * n_timesteps * np.log(2 * np.pi) - 0.5 * l1 - 0.5 * l2
        return -likelihood
