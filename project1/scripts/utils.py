import numpy as np


def calculate_mse(e):
    """Calculate the mse for error vector e."""
    return 1/2 * np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for error vector e."""
    return np.mean(np.abs(e))


def calculate_loss(y, tx, w):
    """Calculate the loss."""
    e = y - tx @ w
    return calculate_mse(e)
    # return calculate_mae(e)


def feature_scale(X):
    """Feature scaling"""
    min_X = np.min(X, axis=0)
    max_X = np.max(X, axis=0)
    X_norm = (X - min_X) / (max_X - min_X)

    return X_norm


def standardize(X):
    """Standardization"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_stand = (X - mean) / std

    return X_stand


def sigmoid(x):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-x))


def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    # pred = sigmoid(tx @ w)
    # loss = y.T @ np.log(pred) + (1 - y).T @ np.log(1 - pred)
    loss = np.sum(np.log(1 + np.exp(tx @ w))) - y.T @ tx @ w
    return loss
