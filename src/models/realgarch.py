import numpy as np
from scipy.optimize import minimize
from statsmodels.api import OLS, add_constant

from .filters import (
    filter_realgarch_symmetric,
    filter_realgarch_asymmetric
)
from .likelihoods import (
    negloglik_realgarch_symmetric,
    negloglik_realgarch_asymmetric,
)
from .base import negloglik_returns

def estimate_realgarch_symmetric_2step(returns, rv5):
    """
    Two-step estimation of symmetric Realized GARCH.
    """

    # Step 1: returns-only likelihood
    init_r = [np.mean(returns), 0.02, 0.05, 0.85, 0.1]
    bounds_r = [
        (-np.inf, np.inf), (1e-6, None),
        (1e-6, 1), (1e-6, 1), (0, 1)
    ]

    res_r = minimize(
        negloglik_returns,
        init_r,
        args=(returns, rv5),
        bounds=bounds_r
    )

    mu, omega, alpha, beta, gamma = res_r.x

    # Step 2: construct sigma² and residuals
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)

    for t in range(1, n):
        sigma2[t] = (
            omega
            + alpha * (returns[t-1] - mu) ** 2
            + beta * sigma2[t-1]
            + gamma * rv5[t-1]
        )
        sigma2[t] = max(sigma2[t], 1e-8)

    eps = (returns - mu) / np.sqrt(sigma2)
    X = add_constant(np.column_stack([sigma2, eps**2 - 1]))

    reg = OLS(rv5, X).fit()
    xi, phi, tau2 = reg.params
    sigma_u = np.std(reg.resid)

    full_params = [mu, omega, alpha, beta, gamma, xi, phi, tau2, sigma_u]

    sigma2, *_ = filter_realgarch_symmetric(full_params, returns, rv5)

    loglik = -negloglik_realgarch_symmetric(full_params, returns, rv5)

    return {
        "params": full_params,
        "sigma2": sigma2,
        "loglik": loglik
    }

def estimate_realgarch_asymmetric_2step(returns, rv5):
    """
    Two-step estimation of asymmetric Realized GARCH.
    """

    init_r = [np.mean(returns), 0.02, 0.05, 0.85, 0.1]
    bounds_r = [
        (-np.inf, np.inf), (1e-6, None),
        (1e-6, 1), (1e-6, 1), (0, 1)
    ]

    res_r = minimize(
        negloglik_returns,
        init_r,
        args=(returns, rv5),
        bounds=bounds_r
    )

    mu, omega, alpha, beta, gamma = res_r.x

    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)

    for t in range(1, n):
        sigma2[t] = (
            omega
            + alpha * (returns[t-1] - mu) ** 2
            + beta * sigma2[t-1]
            + gamma * rv5[t-1]
        )
        sigma2[t] = max(sigma2[t], 1e-8)

    eps = (returns - mu) / np.sqrt(sigma2)

    X = add_constant(np.column_stack([sigma2, eps, eps**2 - 1]))
    reg = OLS(rv5, X).fit()

    xi, phi, tau1, tau2 = reg.params
    sigma_u = np.std(reg.resid)

    full_params = [mu, omega, alpha, beta, gamma, xi, phi, tau1, tau2, sigma_u]

    sigma2, *_ = filter_realgarch_asymmetric(full_params, returns, rv5)

    loglik = -negloglik_realgarch_asymmetric(full_params, returns, rv5)

    return {
        "params": full_params,
        "sigma2": sigma2,
        "loglik": loglik
    }
