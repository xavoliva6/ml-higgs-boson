import numpy as np

from utils import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma, **kwargs):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    N, D = tx.shape
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient
        e = y - tx @ w
        grad = - 1 / N * tx.T @ e

        # update w by gradient descent update
        w -= gamma * grad

    # calculate final loss
    y_pred = tx @ w
    loss = calculate_mse_loss(y, y_pred)

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, **kwargs):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    N, D = tx.shape
    w = initial_w

    for n_iter in range(max_iters):
        # select a random sample
        nr = np.random.randint(low=0, high=N)
        tx_s, y_s = tx[nr], y[nr]
        # compute gradient
        e = y_s - tx_s @ w
        grad = -tx_s.T * e

        # update w by gradient descent update
        w -= gamma * grad

    # calculate loss
    y_pred = tx @ w
    loss = calculate_mse_loss(y, y_pred)

    return w, loss


def least_squares(y, tx, **kwargs):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    N, D = tx.shape
    # compute w using explicit solution
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    # calculate loss
    y_pred = tx @ w
    loss = calculate_mse_loss(y, y_pred)

    return w, loss


def ridge_regression(y, tx, lambda_, **kwargs):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    N, D = tx.shape

    # compute w using explicit solution
    w = np.linalg.solve(tx.T @ tx + lambda_ * np.identity(D), tx.T @ y)

    # calculate loss
    y_pred = tx @ w
    loss = calculate_mse_loss(y, y_pred) + lambda_ * w.T @ w

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, **kwargs):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    N, D = tx.shape

    w = initial_w
    for n_iter in range(max_iters):
        e = y - sigmoid(tx @ w)
        grad_log_regression = -1 / N * tx.T @ e
        # update w by gradient descent update
        w -= gamma * grad_log_regression

    # calculate final loss
    loss = calculate_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    N, D = tx.shape

    w = initial_w
    for n_iter in range(max_iters):
        e = y - sigmoid(tx @ w)
        grad_reg_log_regression = -1 / N * tx.T @ e + 2 * lambda_ * w

        # update w by gradient descent update
        w -= gamma * grad_reg_log_regression

    # calculate final loss
    loss = calculate_logistic_loss(y, tx, w) + lambda_ * w.T @ w

    return w, loss
