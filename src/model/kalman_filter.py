import numpy as np

def kalman_filter(Y, A, C, Q, R, x0, P0):
    """
    Implémentation du filtre de Kalman pour estimer les états.
    
    Arguments:
    Y -- observations (log-prix)
    A -- matrice de transition d'état
    C -- matrice de mesure
    Q -- covariance du bruit de processus
    R -- covariance du bruit de mesure
    x0 -- état initial estimé
    P0 -- covariance de l'état initial
    
    Retourne:
    x_est -- estimations des états
    P_est -- estimations des covariances d'état
    """
    n_timesteps = Y.shape[0]
    n_states = A.shape[0]
    
    # Initialisations
    x_pred = x0
    P_pred = P0
    x_est = np.zeros((n_timesteps, n_states))
    P_est = np.zeros((n_timesteps, n_states, n_states))
    
    for t in range(n_timesteps):
        # Mise à jour
        S = C.dot(P_pred).dot(C.T) + R
        K = P_pred.dot(C.T).dot(np.linalg.inv(S))
        y_pred = C.dot(x_pred)
        x_upd = x_pred + K.dot(Y[t] - y_pred)
        P_upd = P_pred - K.dot(S).dot(K.T)
        
        # Prédiction
        x_pred = A.dot(x_upd)
        P_pred = A.dot(P_upd).dot(A.T) + Q
        
        # Enregistrer les estimations
        x_est[t] = x_upd
        P_est[t] = P_upd
    
    return x_est, P_est


