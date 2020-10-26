# -*- coding: utf-8 -*-
"""
This file contains all implemented regression functions.
"""

import config
import numpy as np

from utils import calculate_mse_loss, sigmoid, calculate_logistic_loss, calculate_hinge_loss, \
    convert_class_labels_logistic

GRAD_STOP_CONDITION = 1e-15


def least_squares_GD(y, tx, initial_w, max_iters, gamma, **kwargs):
    """
    Least squares regression using gradient descent.

    Args:
        y (ndarray): 1D array containing labels
        tx (ndarray): 2D array containing dataset
        initial_w (ndarray): initialized weights
        max_iters (int): maximum number of iterations of the algorithm
        gamma (float): Step size of the gradient descent

    Returns:
        ndarray: final weights
        float: final loss

    Raises:
        ValueError: If the weights get too big
    """

    N, D = tx.shape
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient
        e = y - tx @ w
        grad = - 1 / N * tx.T @ e

        # update w by gradient descent update
        w -= gamma * grad

        if np.max(w) > 1e9:
            raise ValueError('Least Squares GD diverged!!!')

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
        initial_w (ndarray): initialized weights
        max_iters (int): maximum number of iterations of the algorithm
        gamma (float): step size of the gradient descent

    Returns:
        ndarray: final weights
        float: final loss

    Raises:
        ValueError: If the weights get too big
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

        if np.max(w) > 1e9:
            raise ValueError('Least Squares SGD diverged!!!')

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
    # w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    # We used linalg.lstsq instead of linalg.solve, because it is preferred when dealing with poorly conditioned
    # matrices, see the following link:
    # https://stackoverflow.com/questions/41648246/efficient-computation-of-the-least-squares-algorithm-in-numpy
    # When adding rows to equalize class imbalance, tx.T @ tx does not have a full column rank.
    try:
        w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    except:
        w = np.linalg.lstsq(tx, y)
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
    # although this would be the "correct" formula, we will instead not
    #w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.identity(D), tx.T @ y)
    # use 2*N, due to the fact that it makes iterating over a range of lambdas
    # much harder
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
        initial_w (ndarray): initialized weights
        max_iters (int): maximum number of iterations of the algorithm
        gamma (float): step size of the gradient descent

    Returns:
        ndarray: final weights
        float: final loss

    Raises:
        ValueError: If the weights get too big
    """

    N, D = tx.shape
    y = convert_class_labels_logistic(y)
    w = initial_w
    for n_iter in range(max_iters):
        e = y - sigmoid(tx @ w)
        grad_log_regression = -1 / N * tx.T @ e
        # update w by gradient descent update
        w -= gamma * grad_log_regression

        if np.max(w) > 1e9:
            raise ValueError('Logistic Regression diverged!!!')

    # calculate final loss
    loss = calculate_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, **kwargs):
    """
    Regularized logistic regression using gradient descent.

    Args:
        y (ndarray): 1D array containing labels
        tx (ndarray): 2D array containing dataset
        lambda_ (float): Regularization parameter
        initial_w (ndarray): initialized weights
        max_iters (int): maximum number of iterations of the algorithm
        gamma (float): step size of the gradient descent

    Returns:
        ndarray: final weights
        float: final loss

    Raises:
        ValueError: If the weights get too big
    """

    N, D = tx.shape
    y = convert_class_labels_logistic(y)

    w = initial_w
    for n_iter in range(max_iters):
        e = y - sigmoid(tx @ w)
        grad_reg_log_regression = -1 / N * tx.T @ e + 2 * lambda_ * w

        # update w by gradient descent update
        w -= gamma * grad_reg_log_regression

        if np.max(w) > 1e9:
            raise ValueError('Regularized Logistic Regression diverged!!!')

    # calculate final loss
    loss = calculate_logistic_loss(y, tx, w) + lambda_ * w.T @ w

    return w, loss


# Additional algorithms #

def support_vector_machine_GD(y, tx, initial_w, max_iters, gamma, lambda_, **kwargs):
    """
    Support vector regression using gradient descent.

    Args:
        y (ndarray): 1D array containing labels
        tx (ndarray): 2D array containing dataset
        lambda_ (float): Regularization parameter
        initial_w (ndarray): initialized weights
        max_iters (int): maximum number of iterations of the algorithm
        gamma (float): step size of the gradient descent

    Returns:
        ndarray: final weights
        float: final loss

    Raises:
        ValueError: If the weights get too big
    """
    N, D = tx.shape
    w = initial_w
    previous_loss = 0
    for n_iter in range(max_iters):

        grad = np.zeros(shape=w.shape)
        # https://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf
        s_vec = y * (tx @ w)
        # compute gradient
        grad += -y[s_vec < 1] @ tx[s_vec < 1]
        grad = 1 / N * (grad + 2 * lambda_ * w)

        # update w by gradient descent update
        w -= gamma * grad

        loss = calculate_hinge_loss(y, tx, w) + lambda_ * w.T @ w
        if np.abs(loss - previous_loss) < 1e-6:
            break
        else:
            previous_loss = loss

        if np.max(w) > 1e9:
            raise ValueError('Support Vector Machine diverged!!!')

    # calculate final loss
    loss = calculate_hinge_loss(y, tx, w) + lambda_ * w.T @ w

    return w, loss


def least_squares_BGD(y, tx, initial_w, max_iters, gamma, batch_size=1024, **kwargs):
    """
    Least Squares regression using mini-batch gradient descent with diminishing step size.

    Args:
        y (ndarray): 1D array containing labels
        tx (ndarray): 2D array containing dataset
        initial_w (ndarray): initialized weights
        max_iters (int): maximum number of iterations of the algorithm
        gamma (float): step size of the gradient descent
        batch_size (int): size of the batch

    Returns:
        ndarray: final weights
        float: final loss

    Raises:
        ValueError: If the weights get too big
    """

    N, D = tx.shape
    w = initial_w
    n_iter = 1
    norm_gradient = np.inf

    while n_iter <= max_iters and norm_gradient > config.GRAD_STOP_CONDITION:
        # create random batch of batch_size
        perm = np.random.permutation(N)[:batch_size]
        tx_b = tx[perm]
        y_b = y[perm]

        # compute gradient
        e = y_b - tx_b @ w
        grad = - 1 / batch_size * tx_b.T @ e
        # update w by gradient descent update
        w -= gamma / n_iter * grad

        if np.max(w) > 1e9:
            raise ValueError('Least Squares mini-batch GD diverged!!!')

        norm_gradient = np.linalg.norm(grad)
        n_iter += 1

    # calculate loss
    y_pred = tx @ w
    loss = calculate_mse_loss(y, y_pred)

    return w, loss
