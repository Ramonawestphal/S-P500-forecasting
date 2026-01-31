import numpy as np
from scipy.optimize import minimize
from .filters import filter_garch
from .likelihoods import negloglik_garch

def estimate_garch(returns):
    init = [np.mean(returns), 0.02, 0.10, 0.85]
    bounds = [(-np.inf, np.inf), (1e-6, None), (1e-6, 1), (1e-6, 1)]

    res = minimize(negloglik_garch, init, args=(returns,), bounds=bounds)

    sigma2 = filter_garch(res.x, returns)

    return {
        "params": res.x,
        "sigma2": sigma2,
        "loglik": -res.fun
    }
