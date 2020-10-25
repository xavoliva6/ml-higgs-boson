# -*- coding: utf-8 -*-
"""
This file contains auxiliary functions.
"""

import numpy as np


def calculate_acc(y, y_pred):
    """
    Calculate the accuracy of a prediction.

    Args:
        y (ndarray): provided labels
        y_pred (ndarray): predicted labels

    Returns:
        float: accuracy value
    """
    N = y_pred.shape[0]
    y_pred[y_pred < 0] = -1
    y_pred[y_pred >= 0] = 1
    return 1 / N * ((y * y_pred) > 0).sum()


def calculate_mse(e):
    """
    Calculate the mean squared error for error vector e.

    Args:
        e (ndarray): error vector

    Returns:
        float: mean squared error value
    """
    return 1 / 2 * np.mean(e ** 2)


def calculate_mae(e):
    """
    Calculate the mean absolute error for error vector e.

    Args:
        e (ndarray): error vector

    Returns:
        float: mean absolute error value
    """
    return np.mean(np.abs(e))


def calculate_mse_loss(y, y_pred):
    """
    Calculate the MSE loss of a prediction.

    Args:
        y (ndarray): provided labels
        y_pred (ndarray): predicted labels

    Returns:
        float: mean squared error value
    """
    e = y - y_pred
    return calculate_mse(e)


def create_labels(y_regression):
    """
    Create class labels for labels created using regression. Negative values are mapped to -1, while positive
    values are mapped to 0.

    Args:
        y_regression (ndarray): regression labels

    Returns:
        ndarray: class labels
    """
    """Generates class labels given regression labels"""
    y_class = np.zeros_like(y_regression)
    y_class[np.where(y_regression <= 0)] = -1
    y_class[np.where(y_regression > 0)] = 1

    return y_class


def sigmoid(x):
    """
    Numerically stable sigmoid function, as seen in:
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    Sigmoid function is expressed in two different ways, each version covers an
    extreme case: x=inf and x=−inf, respectively

    Args:
        x (ndarray): input of the sigmoid function

    Returns:
        ndarray: output of the sigmoid function
    """
    return np.piecewise(x, [x > 0], [
        lambda i: 1 / (1 + np.exp(-i)),
        lambda i: np.exp(i) / (1 + np.exp(i))
    ])


def convert_class_labels_logistic(y: np.ndarray):
    """
    Convert labels from the values {-1, 1} to {0, 1}

    Args:
        y (ndarray): labels array

    Returns:
        ndarray: labels array with values {0, 1}
    """
    if set(y) == {-1, 1}:
        y = ((y + 1) / 2).astype(int)
    return y


def reconvert_class_labels_logistic(y: np.ndarray):
    """
    Convert labels from the values {0, 1} to {-1, 1}

    Args:
        y (ndarray): labels array

    Returns:
        ndarray: labels array with values {-1, 1}
    """
    if set(y) == {0, 1}:
        y = ((y * 2) - 1).astype(int)
    return y


def calculate_logistic_loss(y, tx, w):
    """
    Calculate the logistic loss.

    Args:
        y (ndarray): provided labels
        tx (ndarray): samples array
        w (ndarray): weights array

    Returns:
        float: logistic loss value
    """
    N = len(y)
    pred = sigmoid(tx @ w)
    # https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
    # https://stackoverflow.com/questions/38125319/python-divide-by-zero-encountered-in-log-logistic-regression
    EPSILON = 1e-10
    loss = -1 / N * (y.T @ np.log(pred + EPSILON) + (1 - y).T @ np.log(1 - pred + EPSILON))

    return loss


def calculate_hinge_loss(y, tx, w):
    """
    Calculate the hinge loss.

    Args:
        y (ndarray): provided labels
        tx (ndarray): samples array
        w (ndarray): weights array

    Returns:
        float: hinge loss value
    """
    N, D = tx.shape
    # https://medium.com/@saishruthi.tn/support-vector-machine-using-numpy-846f83f4183d
    c_x = np.ones(shape=N) - y * (tx @ w)
    c_x = np.clip(c_x, a_min=0, a_max=np.inf)
    return c_x.mean()


def cross_validation_iter(y, x, k_indices, k_iteration):
    """
    Compute cross validation for a single iteration.

    Args:
        y (ndarray): provided labels
        x (ndarray): samples array
        k_indices (ndarray): indices of cross validation of all iterations
        k_iteration (int): index of current cross validation step

    Returns:
        ndarray: training set samples
        ndarray: training set labels
        ndarray: validation set samples
        ndarray: validation set labels
    """
    k = k_indices.shape[0]

    val_index = k_indices[k_iteration]
    train_index = k_indices[np.arange(k) != k_iteration]
    train_index = train_index.reshape(-1)

    y_val = y[val_index]
    y_train = y[train_index]
    x_val = x[val_index]
    x_train = x[train_index]

    return x_train, y_train, x_val, y_val


def build_k_indices(y, k):
    """
    Build k indices for k-fold cross-validation.

    Args:
        y (ndarray): provided labels
        k (int): number of iterations

    Returns:
        ndarray: array of indices of all iterations
    """
    N = len(y)
    fold_interval = int(N / k)
    indices = np.random.permutation(N)
    k_indices = [indices[k_iteration * fold_interval: (k_iteration + 1) * fold_interval] for k_iteration in range(k)]

    return np.array(k_indices)


def transform_log_dict_to_param_dict(log_dict):
    """
    For every group of the dataset, find the function-parameters combination with the highest accuracy
    and create a dictionary with its values.

    Args:
        log_dict (dict): dictionary from logs

    Returns:
        dict: dictionary with function and parameters of the best function-parameters combination for every group
    """
    param_dict = {}
    for group_indx in range(1, 7):
        best_acc_indx = np.argmax([t[0] for t in log_dict[str(group_indx)]])
        best_acc_dict = log_dict[str(group_indx)][best_acc_indx][1]
        param_dict[str(group_indx)] = {
            "M": best_acc_dict["M"],
            "corr_anal": best_acc_dict["CA"],
            "class_eq": best_acc_dict["CE"],
            "z_outlier": best_acc_dict["Z"],
            "function_name": best_acc_dict["function"],
            "params": best_acc_dict["params"]
        }
    return param_dict
