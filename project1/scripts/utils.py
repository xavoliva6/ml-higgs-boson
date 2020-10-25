# -*- coding: utf-8 -*-
"""
This file contains auxiliary functions.
"""

import numpy as np

def calculate_acc(y, y_pred):
    N = y_pred.shape[0]
    y_pred[y_pred < 0] = -1
    y_pred[y_pred >= 0] = 1
    return 1 / N * ((y * y_pred) > 0).sum()


def calculate_mse(e):
    """Calculate the mse for error vector e."""
    return 1 / 2 * np.mean(e ** 2)


def calculate_mae(e):
    """Calculate the mae for error vector e."""
    return np.mean(np.abs(e))


def calculate_mse_loss(y, y_pred):
    """Calculate the loss."""
    e = y - y_pred
    return calculate_mse(e)


def create_labels(y_regression):
    """Generates class labels given regression labels"""
    y_class = np.zeros_like(y_regression)
    y_class[np.where(y_regression <= 0)] = -1
    y_class[np.where(y_regression > 0)] = 1

    return y_class


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.piecewise(x, [x > 0], [
        lambda i: 1 / (1 + np.exp(-i)),
        lambda i: np.exp(i) / (1 + np.exp(i))
    ])


def convert_class_labels_logistic(y: np.ndarray):
    """convert from interval [-1, 1] to [0, 1]"""
    if set(y) == {-1, 1}:
        y = ((y + 1) / 2).astype(int)
    return y


def reconvert_class_labels_logistic(y: np.ndarray):
    """convert from interval [0, 1] to [-1, 1]"""
    if set(y) == {0, 1}:
        y = ((y * 2) - 1).astype(int)
    return y


def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    N = len(y)
    EPSILON = 1e-10
    pred = sigmoid(tx @ w)
    # https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
    # https://stackoverflow.com/questions/38125319/python-divide-by-zero-encountered-in-log-logistic-regression
    loss = -1 / N * (y.T @ np.log(pred + EPSILON) + (1 - y).T @ np.log(1 - pred + EPSILON))

    return loss


def calculate_hinge_loss(y, tx, w):
    """calculate the hinge loss."""
    # https://medium.com/@saishruthi.tn/support-vector-machine-using-numpy-846f83f4183d
    c_x = np.ones(shape=tx.shape[0]) - y * (tx @ w)
    c_x = np.clip(c_x, a_min=0, a_max=np.inf)
    return c_x.mean()


def cross_validation_iter(y, x, k_indices, k_iteration):
    """Compute cross validation for a single iteration."""
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
    """build k indices for k-fold cross-validation."""
    N = len(y)
    fold_interval = int(N / k)
    indices = np.random.permutation(N)
    k_indices = [indices[k_iteration * fold_interval: (k_iteration + 1) * fold_interval] for k_iteration in range(k)]

    return np.array(k_indices)


def transform_log_dict_to_param_dict(log_dict):
    param_dict = {}
    for group_indx in range(1, 7):
        best_acc_indx = np.argmax([t[0] for t in log_dict[str(group_indx)]])
        best_acc_dict = log_dict[str(group_indx)][best_acc_indx][1]
        param_dict[str(group_indx)] = {"M": best_acc_dict["M"],
                                       "corr_anal": best_acc_dict["CA"],
                                       "class_eq": best_acc_dict["CE"],
                                       "z_outlier": best_acc_dict["Z"],
                                       "function_name": best_acc_dict["function"],
                                       "params": best_acc_dict["params"]
                                       }
    return param_dict
