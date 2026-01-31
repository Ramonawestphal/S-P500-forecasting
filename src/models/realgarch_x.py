import numpy as np
from scipy.optimize import minimize
from statsmodels.api import OLS, add_constant

from .filters import filter_realgarch_x
from .likelihoods import negloglik_realgarch_x

def estimate_realgarch_x_2step(returns, rv5_spx, rv5_exo):
    """
    Two-step Realized GARCH-X with external realized volatility.
    """

    def negloglik_returns_x(params):
        mu, omega, alpha, beta, gamma, delta = params
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = (
                omega
                + alpha * (returns[t-1] - mu) ** 2
                + beta * sigma2[t-1]
                + gamma * rv5_spx[t-1]
                + delta * rv5_exo[t-1]
            )
            sigma2[t] = max(sigma2[t], 1e-8)

        eps = returns - mu
        ll = -0.5 * (
            np.log(2 * np.pi)
            + np.log(sigma2)
            + eps**2 / sigma2
        )
        return -np.sum(ll)

    init = [np.mean(returns), 0.02, 0.05, 0.85, 0.1, 0.1]
    bounds = [
        (-np.inf, np.inf), (1e-6, None),
        (1e-6, 1), (1e-6, 1),
        (0, 1), (0, 1)
    ]

    res = minimize(negloglik_returns_x, init, bounds=bounds)

    mu, omega, alpha, beta, gamma, delta = res.x

    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)

    for t in range(1, n):
        sigma2[t] = (
            omega
            + alpha * (returns[t-1] - mu) ** 2
            + beta * sigma2[t-1]
            + gamma * rv5_spx[t-1]
            + delta * rv5_exo[t-1]
        )
        sigma2[t] = max(sigma2[t], 1e-8)

    eps = (returns - mu) / np.sqrt(sigma2)

    X = add_constant(np.column_stack([sigma2, eps, eps**2 - 1]))
    reg = OLS(rv5_spx, X).fit()

    xi, phi, tau1, tau2 = reg.params
    sigma_u = np.std(reg.resid)

    full_params = [mu, omega, alpha, beta, gamma, delta, xi, phi, tau1, tau2, sigma_u]

    sigma2, *_ = filter_realgarch_x(full_params, returns, rv5_spx, rv5_exo)

    loglik = -negloglik_realgarch_x(full_params, returns, rv5_spx, rv5_exo)

    return {
        "params": full_params,
        "sigma2": sigma2,
        "loglik": loglik
    }
