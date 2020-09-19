import numpy as np



def data_replacement(X, method="mean"):
    """
    replaces missin data points in either matrices or vectors

    different methods: mean
    """

    def fill_vec(vec, method):
        if method == "mean":
            # find all missing points
            missing = vec == -999.
            # calculate the remaining points mean
            feature_mean = np.mean(vec[~missing])
            # fill them up
            vec[missing] = feature_mean
            return vec



    # for  a vector
    if X.ndim == 1:
        return fill_vec(X, method=method)

    # for a matrix
    if X.ndim == 2:
        for vec_indx, vec in enumerate(X.T):
            X[:, vec_indx] = fill_vec(vec, method=method)
        return X


def z_score_outlier_detection(X, thresh=2.5):
    """
    Peforms iterative z score outlier detection, in which detect outliers
    are repalced.

    Args:
        X: NxD Matrix, where we look for outliers witihin columns
        thresh: z score threshold

    Returns
        X: NxD Matrix, without outliers
    """

    for f_indx, feature_vec in enumerate(X.T):
        while(True):
            # calculate z scores
            z_scores = ((feature_vec)-np.mean(feature_vec))/np.std(feature_vec)
            # find all z score above threshold
            outliers = np.abs(z_scores)>thresh
            # if there are none, stop with this feature
            if not outliers.any():
                X[:, f_indx] = feature_vec
                break
            # if there are, remove and replace them
            else:
                feature_vec[outliers] = -999.
                feature_vec = data_replacement(feature_vec)
    return X
