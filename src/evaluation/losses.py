import numpy as np


def mse_loss(predictions, targets):
    """
    Mean Squared Error
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return np.mean((predictions - targets) ** 2)


def qlike_loss(predictions, targets, eps=1e-8):
    """
    QLIKE loss (robust to noise in volatility measurement)
    """
    predictions = np.maximum(np.asarray(predictions), eps)
    targets = np.maximum(np.asarray(targets), eps)
    return np.mean(targets / predictions - np.log(targets / predictions) - 1)
