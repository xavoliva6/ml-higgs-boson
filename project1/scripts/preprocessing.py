import numpy as np


<<<<<<< Updated upstream
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


def split_data(X, Y, ids, val_prop=0.3):
    """Splits data into training and validation"""
    # TODO maybe add randomization here?
    X_tr = X[:int(X.shape[0] * (1 - val_prop))]
    Y_tr = Y[:int(X.shape[0] * (1 - val_prop))]
    ids_tr = ids[:int(X.shape[0] * (1 - val_prop))]

    X_val = X[int(X.shape[0] * (1 - val_prop)):]
    Y_val = Y[int(X.shape[0] * (1 - val_prop)):]
    ids_val = ids[int(X.shape[0] * (1 - val_prop)):]

    return (X_tr, Y_tr, ids_tr), (X_val, Y_val, ids_val)

=======
def corr_filter(X, threshold):
    n = X.shape[1]
    columns = np.ones((n,))
    for i in range(n-1):
        for j in range(i+1, n):
            if columns[j] == 1:
                correlation = np.abs(np.corrcoef(X[:,i], X[:,j])[0,1])
                if correlation >= threshold:
                    columns[j] = 0
    return columns
    
>>>>>>> Stashed changes

def data_replacement(X, method="median"):
    """
    Replaces missing data points in either matrices or vectors
    different methods: mean
    """

    # for  a vector
    if X.ndim == 1:
        return fill_vec(X, method=method)

    # for a matrix
    if X.ndim == 2:
        for vec_index, vec in enumerate(X.T):
            X[:, vec_index] = fill_vec(vec, method=method)
        return X


def fill_vec(vec, method):
    # find all missing points
    missing = vec == -999.
    feature_method = 0

    if method == "mean":
        # calculate the remaining points mean
        feature_method = np.mean(vec[~missing])
    elif method == "median":
        # calculate the remaining points median
        feature_method = np.median(vec[~missing])
    else:
        print("Warning: Method not implemented")

    # fill them up
    vec[missing] = feature_method

    return vec


def z_score_outlier_detection(X, thresh=2.5):
    """
    Performs iterative z score outlier detection, in which detect outliers
    are replaced.

    Args:
        X: NxD Matrix, where we look for outliers withing columns
        thresh: z score threshold

    Returns
        X: NxD Matrix, without outliers
    """

    for f_index, feature_vec in enumerate(X.T):
        while True:
            # calculate z scores
            z_scores = (feature_vec - np.mean(feature_vec)) / np.std(feature_vec)
            # find all z score above threshold
            outliers = np.abs(z_scores) > thresh
            # if there are none, stop with this feature
            if not outliers.any():
                X[:, f_index] = feature_vec
                break
            # if there are, remove and replace them
            else:
                feature_vec[outliers] = -999.
                feature_vec = data_replacement(feature_vec)

    return X


def add_ones_column(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def augment_features_polynomial(X, M):
    """Augment the input with a polynomial basis (of arbitrary degree M )"""
    if M < 2:
        return X

    index_start = 1 if np.all(X[:, 0] == 1) else 0

    X_poly = X

    for i in range(2, M + 1):
        X_powered = np.power(X[:, index_start:], i)
        X_poly = np.hstack((X_poly, X_powered))

    return X_poly
