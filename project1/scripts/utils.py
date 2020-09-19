import numpy as np


def calculate_mse(e):
    """Calculate the mse for error vector e."""
    return 1/2 * np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for error vector e."""
    return np.mean(np.abs(e))


def calculate_mse_loss(y, y_pred):
    """Calculate the loss."""
    e = y - y_pred
    return calculate_mse(e)


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


def cross_validation(y, x, k_indices, k_iteration):
    """Compute cross validation for specific iteration."""
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
    N = y.shape[0]
    fold_interval = int(N / k)
    indices = np.random.permutation(N)
    k_indices = [indices[k_iteration * fold_interval: (k_iteration+1) * fold_interval] for k_iteration in range(k)]

    return np.array(k_indices)
