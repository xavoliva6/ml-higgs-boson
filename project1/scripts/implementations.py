import numpy as np

from utils import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    N, D = tx.shape
    w = initial_w

    # add column of ones to add bias term
    tx = np.hstack((np.ones((N, 1)), tx))

    for n_iter in range(max_iters):
        # compute gradient
        e = y - tx @ w
        grad = - 1/N * tx.T @ e

        # update w by gradient descent update
        w = w - gamma * grad

    # calculate final loss
    loss = calculate_loss(y, tx, w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    N, D = tx.shape
    w = initial_w

    # add column of ones to add bias term
    tx = np.hstack((np.ones((N, 1)), tx))

    for n_iter in range(max_iters):
        # select a random sample
        nr = np.random.randint(low=0, high=N)
        tx_s, y_s = tx[nr], y[nr]
        # compute gradient
        err = y_s - tx_s @ w
        grad = -tx_s.T * err

        # update w by gradient descent update
        w = w - gamma * grad

    # calculate loss
    loss = calculate_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    N, D = tx.shape

    # add column of ones to add bias term
    tx = np.hstack((np.ones((N, 1)), tx))

    # compute w using explicit solution
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y

    # calculate loss
    loss = calculate_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    N, D = tx.shape

    # add column of ones to add bias term
    tx = np.hstack((np.ones((N, 1)), tx))

    # compute w using explicit solution
    w = np.linalg.inv(tx.T @ tx + lambda_/2*N * np.identity(D+1)) @ tx.T @ y

    # calculate loss
    loss = calculate_loss(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    N, D = tx.shape

    # add column of ones to add bias term
    tx = np.hstack((np.ones((N, 1)), tx))

    w = initial_w
    for n_iter in range(max_iters):

        grad_log_loss = tx.T @ (sigmoid(tx @ w) - y)

        # update w by gradient descent update
        w = w - gamma * grad_log_loss

    # calculate final loss
    loss = calculate_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    N, D = tx.shape

    # add column of ones to add bias term
    tx = np.hstack((np.ones((N, 1)), tx))

    w = initial_w
    for n_iter in range(max_iters):
        grad_reg_log_loss = 1/N * tx.T @ (sigmoid(tx @ w) - y) + (lambda_ / N) * w

        # update w by gradient descent update
        w = w - gamma * grad_reg_log_loss

    # calculate final loss
    loss = calculate_logistic_loss(y, tx, w)

    return w, loss
