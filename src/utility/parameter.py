def calculate_num_parameters(n_factors):
    """Calculate the number of parameters based on the number of factors, including initial state for non-stationary variable."""
    return int(0.5 * n_factors**2 + 2.5 * n_factors )

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


