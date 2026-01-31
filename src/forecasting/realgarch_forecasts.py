import numpy as np

def forecast_realgarch_symmetric(params, returns, rv5, horizon):
    """
    d-step ahead variance forecast for Symmetric Realized GARCH
    """
    mu, omega, alpha, beta, gamma, xi, phi, tau2, _ = params

    n = len(returns)
    sigma2 = np.zeros(n)

    unconditional = (omega + gamma * xi) / max(1 - alpha - beta - gamma * phi, 1e-6)
    sigma2[0] = unconditional

    for t in range(1, n):
        sigma2[t] = (
            omega
            + alpha * (returns[t-1] - mu) ** 2
            + beta * sigma2[t-1]
            + gamma * rv5[t-1]
        )
        sigma2[t] = max(sigma2[t], 1e-8)

    if horizon == 1:
        return sigma2[-1]

    persistence = alpha + beta + gamma * phi
    return unconditional + persistence ** (horizon - 1) * (sigma2[-1] - unconditional)


def forecast_realgarch_asymmetric(params, returns, rv5, horizon):
    """
    d-step ahead variance forecast for Asymmetric Realized GARCH
    """
    mu, omega, alpha, beta, gamma, xi, phi, tau1, tau2, _ = params

    n = len(returns)
    sigma2 = np.zeros(n)

    unconditional = (omega + gamma * xi) / max(1 - alpha - beta - gamma * phi, 1e-6)
    sigma2[0] = unconditional

    for t in range(1, n):
        sigma2[t] = (
            omega
            + alpha * (returns[t-1] - mu) ** 2
            + beta * sigma2[t-1]
            + gamma * rv5[t-1]
        )
        sigma2[t] = max(sigma2[t], 1e-8)

    if horizon == 1:
        return sigma2[-1]

    persistence = alpha + beta + gamma * phi
    return unconditional + persistence ** (horizon - 1) * (sigma2[-1] - unconditional)