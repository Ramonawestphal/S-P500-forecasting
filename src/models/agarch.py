import numpy as np
from scipy.optimize import minimize

from .filters import filter_agarch
from .likelihoods import negloglik_agarch


def estimate_agarch(returns):
    """
    Estimate an asymmetric GARCH(1,1) model.
    """

    init = [np.mean(returns), 0.02, 0.10, 0.10, 0.85]
    bounds = [
        (-np.inf, np.inf),   # mu
        (1e-6, None),        # omega
        (0, 1),              # alpha1
        (0, 1),              # alpha2
        (1e-6, 1)            # beta
    ]

    res = minimize(
        negloglik_agarch,
        init,
        args=(returns,),
        bounds=bounds
    )

    sigma2 = filter_agarch(res.x, returns)

    return {
        "params": res.x,
        "sigma2": sigma2,
        "loglik": -res.fun
    }
