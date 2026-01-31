import numpy as np


def forecast_har(beta, rv_d, rv_w, rv_m):
    """
    Direct HAR-RV forecast (already horizon-specific)
    """
    return beta[0] + beta[1]*rv_d + beta[2]*rv_w + beta[3]*rv_m
