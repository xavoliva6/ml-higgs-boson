import numpy as np

from utils import calculate_mse


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        err = y - tx @ w
        grad = -tx.T @ err / len(err)

        # update w by gradient descent update
        w = w - gamma * grad

    # calculate loss
    err = y - tx @ w
    loss = calculate_mse(err)

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    pass


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    # add column of ones to add bias term
    tx = np.hstack((np.ones((tx.shape[0], 1)), tx))

    # compute w using pseudo-inverse
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y

    # calculate loss
    err = y - tx @ w
    loss = calculate_mse(err)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    pass


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    pass


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    pass