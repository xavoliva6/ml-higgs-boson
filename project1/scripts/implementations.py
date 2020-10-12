import numpy as np

from utils import calculate_mse_loss, sigmoid, calculate_logistic_loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma, **kwargs):
    """
    Linear regression using gradient descent.

    Args:
        y (ndarray): 1D array containing labels
        tx (ndarray): 2D array containing dataset
        initial_w (type): initialized weights
        max_iters (int): maximum number of iterations of the algorithm
        gamma (float): Step size of the gradient descent

    Returns:
        ndarray: final weights
        float: final loss

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
    Linear regression using stochastic gradient descent.

    Args:
        y (ndarray): 1D array containing labels
        tx (ndarray): 2D array containing dataset
        initial_w (type): initialized weights
        max_iters (int): maximum number of iterations of the algorithm
        gamma (float): step size of the gradient descent

    Returns:
        ndarray: final weights
        float: final loss

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
    Least squares regression computed using the pseudo-inverse.

    Args:
        y (ndarray): 1D array containing labels
        tx (ndarray): 2D array containing dataset

    Returns:
        ndarray: final weights
        float: final loss

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
    Ridge regression computed using the regularized pseudo-inverse.

    Args:
        y (ndarray): 1D array containing labels
        tx (ndarray): 2D array containing dataset
        lambda_ (float): regularization parameter

    Returns:
        ndarray: final weights
        float: final loss

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
    Logistic regression using gradient descent.

    Args:
        y (ndarray): 1D array containing labels
        tx (ndarray): 2D array containing dataset
        initial_w (type): initialized weights
        max_iters (int): maximum number of iterations of the algorithm
        gamma (float): step size of the gradient descent

    Returns:
        ndarray: final weights
        float: final loss

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
    Regularized logistic regression using gradient descent.

    Args:
        y (ndarray): 1D array containing labels
        tx (ndarray): 2D array containing dataset
        lambda_ (float): Regularization parameter
        initial_w (type): initialized weights
        max_iters (int): maximum number of iterations of the algorithm
        gamma (float): step size of the gradient descent

    Returns:
        ndarray: final weights
        float: final loss

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
