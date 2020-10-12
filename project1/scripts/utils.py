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


def sigmoid(x):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-x))


def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    N = len(y)
    epsilon = 1e-5
    pred = sigmoid(tx @ w)
    # https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
    # https://stackoverflow.com/questions/38125319/python-divide-by-zero-encountered-in-log-logistic-regression
    loss = -1 / N * (y.T @ np.log(pred + epsilon) + (1 - y).T @ np.log(1 - pred + epsilon))

    return loss


def calculate_hinge_loss(y, tx, w):
    """calculate the hinge loss."""
    # https://medium.com/@saishruthi.tn/support-vector-machine-using-numpy-846f83f4183d
    c_x = np.ones(shape=tx.shape[0]) - y * (tx @ w)
    c_x = np.clip(c_x, a_min=0, a_max=np.inf)
    return c_x.mean()


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
    N = len(y)
    fold_interval = int(N / k)
    indices = np.random.permutation(N)
    k_indices = [indices[k_iteration * fold_interval: (k_iteration + 1) * fold_interval] for k_iteration in range(k)]

    return np.array(k_indices)
