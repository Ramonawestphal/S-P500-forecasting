import numpy as np


def forecast_garch(params, returns, horizon):
    """
    d-step ahead variance forecast for GARCH(1,1)
    """
    mu, omega, alpha, beta = map(float, params)
    n = len(returns)

    sigma2 = np.zeros(n)
    sigma2[0] = omega / max(1 - alpha - beta, 1e-6)

    for t in range(1, n):
        eps = returns[t-1] - mu
        sigma2[t] = omega + alpha * eps**2 + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], 1e-8)

    if horizon == 1:
        return sigma2[-1]

    unconditional = omega / (1 - alpha - beta)
    return unconditional + (alpha + beta)**(horizon-1) * (sigma2[-1] - unconditional)


def forecast_agarch(params, returns, horizon):
    """
    d-step ahead variance forecast for AGARCH
    """
    mu, omega, alpha1, alpha2, beta = map(float, params)
    n = len(returns)

    sigma2 = np.zeros(n)
    alpha_bar = 0.5 * (alpha1 + alpha2)
    sigma2[0] = omega / max(1 - alpha_bar - beta, 1e-6)

    for t in range(1, n):
        r = returns[t-1] - mu
        alpha = alpha1 if r < 0 else alpha2
        sigma2[t] = omega + alpha * r**2 + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], 1e-8)

    if horizon == 1:
        return sigma2[-1]

    unconditional = omega / (1 - alpha_bar - beta)
    return unconditional + (alpha_bar + beta)**(horizon-1) * (sigma2[-1] - unconditional)
