import numpy as np

from utils import *


def least_squares_GD(args):
    """Linear regression using gradient descent"""
    tx, y, initial_w, max_iters, gamma = args["X"], args["Y"], args["W"] ,args["m"], args["g"]

    N, D = tx.shape
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient
        e = y - tx @ w
        grad = - 1/N * tx.T @ e

        # update w by gradient descent update
        w = w - gamma * grad

    # calculate final loss
    y_pred = tx @ w
    loss = calculate_mse_loss(y, y_pred)

    return w, loss


def least_squares_SGD(args):
    """Linear regression using stochastic gradient descent"""
    tx, y, initial_w, max_iters, gamma = args["X"], args["Y"], args["W"] ,args["m"], args["g"]

    N, D = tx.shape
    w = initial_w

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
    y_pred = tx @ w
    loss = calculate_mse_loss(y, y_pred)

    return w, loss


def least_squares(args):
    """Least squares regression using normal equations"""
    tx, y = args["X"], args["Y"]
    N, D = tx.shape

    # compute w using explicit solution
    try:
        w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    except:
        print("!!!! SINGULAR MATRIX IN LEAST SQUARES, DOING PINV")
        w = np.linalg.pinv(tx.T @ tx) @ tx.T @ y
    # calculate loss
    y_pred = tx @ w
    loss = calculate_mse_loss(y, y_pred)

    return w, loss


def ridge_regression(args):
    """Ridge regression using normal equations"""
    tx, y, lambda_ = args["X"], args["Y"], args["l"]
    N, D = tx.shape

    # compute w using explicit solution
    w = np.linalg.inv(tx.T @ tx + lambda_/2*N * np.identity(D)) @ tx.T @ y

    # calculate loss
    y_pred = tx @ w
    loss = calculate_mse_loss(y, y_pred)

    return w, loss


def logistic_regression(args):
    """Logistic regression using gradient descent or SGD"""
    tx, y, initial_w, max_iters, gamma = args["X"], args["Y"], args["W"] ,args["m"], args["g"]
    N, D = tx.shape

    w = initial_w
    for n_iter in range(max_iters):

        grad_log_loss = tx.T @ (sigmoid(tx @ w) - y)

        # update w by gradient descent update
        w = w - gamma * grad_log_loss

    # calculate final loss
    loss = calculate_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(args):
    """Regularized logistic regression using gradient descent or SGD"""
    tx, y, initial_w, max_iters, gamma, lambda_ = args["X"], args["Y"], args["W"] ,args["m"], args["g"], args["l"]
    N, D = tx.shape

    w = initial_w
    for n_iter in range(max_iters):
        grad_reg_log_loss = 1/N * tx.T @ (sigmoid(tx @ w) - y) + (lambda_ / N) * w

        # update w by gradient descent update
        w = w - gamma * grad_reg_log_loss

    # calculate final loss
    loss = calculate_logistic_loss(y, tx, w)

    return w, loss
