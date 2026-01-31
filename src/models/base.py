import numpy as np

def logpdf_normal(y, mu, sigma2):
    """Log PDF of normal distribution (element-wise)."""
    sigma2 = np.maximum(sigma2, 1e-8)
    return -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (y - mu)**2 / sigma2)

# Helper for two-step RealGARCH estimation

def negloglik_returns(params_r, returns, rv5=None):
    mu, omega, alpha, beta, gamma = params_r
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)

    for t in range(1, n):
        sigma2[t] = omega + alpha * (returns[t - 1] - mu) ** 2 + beta * sigma2[t - 1]
        if rv5 is not None:
            sigma2[t] += gamma * rv5[t - 1]
        sigma2[t] = max(sigma2[t], 1e-8)

    epsilon = returns - mu
    loglik_r = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (epsilon ** 2) / sigma2)

    return -np.sum(loglik_r)