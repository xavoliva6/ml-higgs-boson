import numpy as np


def data_replacement(X, method="median"):
    """
    replaces missing data points in either matrices or vectors
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
