def calculate_num_parameters(n_factors):
    if n_factors == 1:
        return 3  # [mu, sigma1, lambda1]
    elif n_factors == 2:
        return 7  # [mu, sigma1, lambda1, kappa2, sigma2, lambda2, rho12]
    elif n_factors == 3:
        return 12  # [mu, sigma1, lambda1, kappa2, sigma2, lambda2, rho12, kappa3, sigma3, lambda3, rho13, rho23]
    elif n_factors == 4:
        return 18  # [mu, sigma1, lambda1, kappa2, sigma2, lambda2, rho12, kappa3, sigma3, lambda3, rho13, rho23, kappa4, sigma4, lambda4, rho14, rho24, rho34]
    else:
        raise ValueError("Unsupported number of factors")

## Constant

DELTA = 0.004

# Sorensen (2002) parameters estimates

mu = 0.0416
kappa = 0.7744
sigma1 = 0.1585
sigma2 = 0.2201
rho = -0.3116
alpha = -0.0386
lambdaz= -0.1011
x1 = 4.8738
sigma_e = 0.0171

initial_guesses = {
    1: [mu, sigma1, lambdaz],
    2: [mu, sigma1, lambdaz, kappa, sigma2, lambdaz, rho],
    3: [mu, sigma1, lambdaz, kappa, sigma2, lambdaz, rho, kappa, sigma2, lambdaz, rho, rho],
    4: [mu, sigma1, lambdaz, kappa, sigma2, lambdaz, rho, kappa, sigma2, lambdaz, rho, rho, kappa, sigma2, lambdaz, rho, rho, rho]
}