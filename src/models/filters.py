from .base import logpdf_normal
import numpy as np

def filter_garch(parameter_vector, returns):
    """ GARCH(1,1) conditional variance filter. """
    n = len(returns)
    mu, omega, alpha, beta = parameter_vector
    sigma2 = np.zeros(n)

    for t in range(n):
        if t == 0:
            denom = max(1 - alpha - beta, 1e-6)
            sigma2[t] = omega / denom
        else:
            residual = returns[t - 1] - mu
            sigma2[t] = omega + alpha * residual ** 2 + beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-8)  # numerical stability

    return sigma2

def filter_agarch(parameter_vector, returns):
    """ Asymmetric GARCH filter """
    n = len(returns)
    mu, omega, alpha1, alpha2, beta = parameter_vector
    sigma2 = np.zeros(n)
    alpha_avg = 0.5 * (alpha1 + alpha2)
    sigma2[0] = omega / (1 - alpha_avg - beta)

    for t in range(1, n):
        r_prev = returns[t - 1] - mu
        alpha = alpha1 if r_prev < 0 else alpha2
        sigma2[t] = omega + alpha * r_prev ** 2 + beta * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], 1e-8)

    return sigma2

def filter_realgarch_symmetric(parameter_vector, returns, rv5):
    n = len(returns)
    mu, omega, alpha, beta, gamma, xi, phi, tau2, sigma_u = parameter_vector

    sigma2 = np.zeros(n)
    epsilon = np.zeros(n)
    u = np.zeros(n)
    loglik_r = np.zeros(n)
    loglik_x = np.zeros(n)

    sigma2_unconditional = max((omega + gamma * xi) / (1 - alpha - beta - gamma * phi), 1e-6)
    sigma2[0] = sigma2_unconditional

    for t in range(1, n):
        sigma2[t] = omega + alpha * (returns[t-1] - mu) ** 2 + beta * sigma2[t-1] + gamma * rv5[t-1]
        sigma2[t] = max(sigma2[t], 1e-8)
        epsilon[t-1] = (returns[t-1] - mu) / np.sqrt(sigma2[t-1])

    for t in range(n):
        loglik_r[t] = logpdf_normal(returns[t], mu, sigma2[t])
        if t > 0:
            u[t-1] = rv5[t-1] - xi - phi * sigma2[t-1] - tau2 * (epsilon[t-1] ** 2 - 1)
            loglik_x[t-1] = logpdf_normal(u[t-1], 0, sigma_u ** 2)

    return sigma2, u, loglik_r, loglik_x

def filter_realgarch_asymmetric(parameter_vector, returns, rv5):
    n = len(returns)
    mu, omega, alpha, beta, gamma, xi, phi, tau1, tau2, sigma_u = parameter_vector

    sigma2 = np.zeros(n)
    epsilon = np.zeros(n)
    u = np.zeros(n)
    loglik_r = np.zeros(n)
    loglik_x = np.zeros(n)

    sigma2_unconditional = max((omega + gamma * xi) / (1 - alpha - beta - gamma * phi), 1e-6)
    sigma2[0] = sigma2_unconditional

    for t in range(1, n):
        sigma2[t] = omega + alpha * (returns[t-1] - mu) ** 2 + beta * sigma2[t-1] + gamma * rv5[t-1]
        sigma2[t] = max(sigma2[t], 1e-8)
        epsilon[t-1] = (returns[t-1] - mu) / np.sqrt(sigma2[t-1])

    for t in range(n):
        loglik_r[t] = logpdf_normal(returns[t], mu, sigma2[t])
        if t > 0:
            u[t-1] = rv5[t-1] - xi - phi * sigma2[t-1] - tau1 * epsilon[t-1] - tau2 * (epsilon[t-1] ** 2 - 1)
            loglik_x[t-1] = logpdf_normal(u[t-1], 0, sigma_u ** 2)

    return sigma2, u, loglik_r, loglik_x

def filter_realgarch_x(parameter_vector, returns, rv5_spx, rv5_exo):
    """
    RealGARCH-X filter with asymmetric measurement equation.
    Includes external realized volatility (rv5_exo) from another index.
    """
    n = len(returns)
    
    # Unpack parameters
    mu, omega, alpha, beta, gamma, delta, xi, phi, tau1, tau2, sigma_u = parameter_vector

    sigma2 = np.zeros(n)
    epsilon = np.zeros(n)
    u = np.zeros(n)
    loglik_r = np.zeros(n)
    loglik_x = np.zeros(n)

    # Initial variance
    sigma2_unconditional = max((omega + gamma * np.mean(rv5_spx) + delta * np.mean(rv5_exo)) / 
                               (1 - alpha - beta - gamma * phi), 1e-6)
    sigma2[0] = sigma2_unconditional

    # Filtering
    for t in range(1, n):
        sigma2[t] = (
            omega +
            alpha * (returns[t-1] - mu)**2 +
            beta * sigma2[t-1] +
            gamma * rv5_spx[t-1] +
            delta * rv5_exo[t-1]
        )
        sigma2[t] = max(sigma2[t], 1e-8)
        epsilon[t-1] = (returns[t-1] - mu) / np.sqrt(sigma2[t-1])

    # Likelihood
    for t in range(n):
        loglik_r[t] = logpdf_normal(returns[t], mu, sigma2[t])
        if t > 0:
            u[t-1] = (
                rv5_spx[t-1]
                - xi
                - phi * sigma2[t-1]
                - tau1 * epsilon[t-1]
                - tau2 * (epsilon[t-1]**2 - 1)
            )
            loglik_x[t-1] = logpdf_normal(u[t-1], 0, sigma_u**2)

    return sigma2, u, loglik_r, loglik_x
