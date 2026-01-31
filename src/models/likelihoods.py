from .filters import (
    filter_garch,
    filter_agarch,
    filter_realgarch_symmetric,
    filter_realgarch_asymmetric,
    filter_realgarch_x
)
from .base import logpdf_normal
import numpy as np

def negloglik_garch(parameter_vector, returns):
    """
    Computes the negative log-likelihood of a GARCH(1,1) model.
    """
    mu = parameter_vector[0]
    sigma2 = filter_garch(parameter_vector, returns)
    log_likelihood = logpdf_normal(returns, mu, sigma2)
    return -np.sum(log_likelihood)

def negloglik_agarch(parameter_vector, returns):
    """
    Computes the negative log-likelihood of a asymmetric GARCH
    """
    mu = parameter_vector[0]
    sigma2 = filter_agarch(parameter_vector, returns)
    log_likelihood = logpdf_normal(returns, mu, sigma2)
    return -np.sum(log_likelihood)

def negloglik_realgarch_symmetric(params, returns, rv5):
    sigma2, u, loglik_r, loglik_x = filter_realgarch_symmetric(params, returns, rv5)
    loglik = np.sum(loglik_r) + np.sum(loglik_x)
    return -loglik

def negloglik_realgarch_asymmetric(params, returns, rv5):
    sigma2, u, loglik_r, loglik_x = filter_realgarch_asymmetric(params, returns, rv5)
    loglik = np.sum(loglik_r) + np.sum(loglik_x)
    return -loglik

def negloglik_realgarch_x(params, returns, rv5_spx, rv5_exo):
    sigma2, u, loglik_r, loglik_x = filter_realgarch_x(params, returns, rv5_spx, rv5_exo)
    loglik = np.sum(loglik_r) + np.sum(loglik_x)
    return -loglik








