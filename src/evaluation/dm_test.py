import numpy as np
from scipy.stats import norm


def diebold_mariano(loss_1, loss_2, h=1):
    """
    Diebold–Mariano test for equal predictive accuracy.

    Parameters
    ----------
    loss_1, loss_2 : array-like
        Loss series (same length)
    h : int
        Forecast horizon

    Returns
    -------
    dm_stat : float
    p_value : float (two-sided)
    """
    loss_1 = np.asarray(loss_1)
    loss_2 = np.asarray(loss_2)

    if loss_1.shape != loss_2.shape:
        raise ValueError("Loss series must have same length")

    d = loss_1 - loss_2
    T = len(d)

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    dm_stat = mean_d / np.sqrt(var_d / T)

    # Two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return dm_stat, p_value
