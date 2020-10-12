import numpy as np
import math

def SMOTE(X, Y, k=4): # TODO takes like 2 hours for 15% of the data... IDEA: Use PCA to find closest neighbours?
    """
    Synthetic Minority Oversampling Technique.

    Determines underpresented class (we assume that there are only 2 classes)
    and creates synthetic samples in order to equalize the imbalance.

    Args:
        X (nd.array): sample matrix [NxD]
        y (nd.array): labels/classes [N]
        k (int):

    Returns:
        nd.array: equalized sample matrix [(N+T)xD]
        nd.array: description [N+T]
    """
    N, D = X.shape

    # find all classes and their abosulte frequency
    abs_freq = {c:sum(Y==c) for c in np.unique(Y)}
    # identify values
    over_represented = max(abs_freq, key=abs_freq.get)
    under_represented = min(abs_freq, key=abs_freq.get)
    under_represented_matrix = X[Y==under_represented]

    # calculate their ratio
    ratio = abs_freq[over_represented]/abs_freq[under_represented]

    # calculate the amount of added samples
    added_samples = int(abs_freq[under_represented]*(ratio-1))
    X_new = np.zeros((N+added_samples, D))
    Y_new = np.zeros(N+added_samples)

    # copy the existing data points
    X_new[:N] = X
    Y_new[:N] = Y

    # create distance matrix
    dist = np.zeros((under_represented_matrix.shape[0], under_represented_matrix.shape[0]))

    print("Calculating Distance Matrix")
    # calculate distance matrix
    for indx1 in range(under_represented_matrix.shape[0]):
        for indx2 in range(indx1):
            print(indx1, under_represented_matrix.shape)
            distance = np.linalg.norm(under_represented_matrix[indx1] - under_represented_matrix[indx1])
            dist[indx1, indx2] = distance
            dist[indx2, indx1] = distance
    print("Finished")

    # get K closest neighbours for each sample
    closest_neighbors = np.zeros((under_represented_matrix.shape[0], k))
    for indx in range(under_represented_matrix.shape[0]):
        closest_neighbors[indx] = np.argsort(dist[indx], 0)[:k]

    # now generate new samples
    for indx in range(added_samples):
        random_indx = np.random.randint(under_represented_matrix.shape[0])
        linear_combination_weights = np.random.dirichlet(np.ones(k),size=1)[0]
        X_new[N+indx] = np.sum([v*w for v,w in zip(under_represented_matrix[closest_neighbors[random_indx].astype(int)], linear_combination_weights)], axis=0)
    # set labels in Y
    Y_new[N:] = under_represented

    return X_new, Y_new


def remove_redundant(X):
    """
    Removes redundant features (i.e. no variance).

    Goes through all columns of a given numpy array and removes all
    redundant features (in the sense that the feature value is constant).

    Args:
        X (nd.array): sample matrix [NxD]

    Returns:
        nd.array: sample matrix without redundant features [Nx(D-T)]
    """
    non_redundant_indc = []
    for col in range(X.shape[1]):
        # check if columns variance is not 0
        if np.var(X[:,col]) != 0:
            non_redundant_indc.append(col)
        else:
            print("FOUND REDUNDANT", col)
    return X[:, non_redundant_indc]


def split_groups(Y, X, group_col_list = [list(range(30)),
                    [k for k in range(30) if k not in [0]],
                    [k for k in range(30) if k not in [4, 5, 6, 12, 26, 27, 28]],
                    [k for k in range(30) if k not in [0, 4, 5, 6, 12, 26, 27, 28]],
                    [k for k in range(30) if k not in [4, 5, 6, 12, 23, 24, 25, 26, 27, 28]],
                    [k for k in range(30) if k not in [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28]]]):
    """
    Splits original sample matrix into subgroups in respect to their missing
    values.

    There are multiple subgroups of missing data, in the sense, that values
    that miss certain values will also miss certain other values. This function
    will split the original sample matrix into smaller subgroups, depending
    on said missing value groups.

    Args:
        Y (nd.array): labels
        X (nd.array): sample matrix
        group_col_list (list): list of the non missing features for each group

    Returns:
        nd.array: list of label array, depending on missing values
        nd.array: list of sample array, depending on missing values
        nd.array: list of row indexes, belonging to the groups of missing values
    """

    # get groups depending on the missing values
    G1 = np.logical_and((X[:,0] != -999.), (X[:,4] != -999.))
    G2 = np.logical_and((X[:,0] == -999.), (X[:,4] != -999.))
    G3 = np.logical_and(np.logical_and((X[:,4] == -999.), (X[:,23] != -999.)), (X[:,0] != -999.))
    G4 = np.logical_and(np.logical_and((X[:,4] == -999.), (X[:,23] != -999.)), (X[:,0] == -999.))
    G5 = np.logical_and(np.logical_and((X[:,4] == -999.), (X[:,23] == -999.)), (X[:,0] != -999.))
    G6 = np.logical_and(np.logical_and((X[:,4] == -999.), (X[:,23] == -999.)), (X[:,0] == -999.))

    group_row_list = [G1, G2, G3, G4, G5, G6]

    # created sample and label subgroups, depending on the gruops
    groups_Y = [Y[indc] for indc in group_row_list]
    groups_X = [remove_redundant(X[indc][:,group_col_list[group]]) for group, indc in enumerate(group_row_list)]

    return groups_Y, groups_X, group_row_list


def feature_scale(X):
    """Feature scaling"""
    # TODO not used?
    min_X = np.min(X, axis=0)
    max_X = np.max(X, axis=0)
    X_norm = (X - min_X) / (max_X - min_X)

    return X_norm


def standardize(X):
    """
    Standardizing an array in respect to columns.

    Each columns values are transform, such that its mean is 0 and variance is
    1.

    Args:
        X (nd.array): array to be standarized

    Returns:
        nd.array: standardized array
    """

    X_stand = np.ones(shape=X.shape)
    index_start = 1 if np.all(X[:, 0] == 1) else 0

    mean = np.mean(X[:, index_start:], axis=0)
    std = np.std(X[:, index_start:], axis=0)

    X_stand[:, index_start:] = (X[:, index_start:] - mean) / std

    return X_stand


def split_data(X, Y, ids, val_prop=0.3):
    """Splits data into training and validation"""
    # TODO maybe add randomization here?
    # TODO not used?
    X_tr = X[:int(X.shape[0] * (1 - val_prop))]
    Y_tr = Y[:int(X.shape[0] * (1 - val_prop))]
    ids_tr = ids[:int(X.shape[0] * (1 - val_prop))]

    X_val = X[int(X.shape[0] * (1 - val_prop)):]
    Y_val = Y[int(X.shape[0] * (1 - val_prop)):]
    ids_val = ids[int(X.shape[0] * (1 - val_prop)):]

    return (X_tr, Y_tr, ids_tr), (X_val, Y_val, ids_val)


def corr_filter(X, threshold):
    """ # TODO ADD DESCRIPTION / What does it do, given we use the subgroups
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    D = X.shape[1]
    columns_to_keep = np.ones(shape=(D,), dtype=bool)

    for i in range(D - 1):
        for j in range(i + 1, D):
            if columns_to_keep[j]:
                correlation = np.abs(np.corrcoef(X[:, i], X[:, j])[0, 1])
                if correlation >= threshold:
                    columns_to_keep[j] = False

    return X[:, columns_to_keep], columns_to_keep


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


def z_score_outlier_detection(X, thresh=2.5): # TODO test this please, do we nee dthis? if not remove the upper 2 functions as well...
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
                feature_vec[outliers] = np.median(feature_vec[~outliers])
                feature_vec = data_replacement(feature_vec)

    return X


def add_bias(X):
    """
    Adds a bias vector as the first column to a given matrix.

    Args:
        X (nd.array): array [NxD]

    Returns:
        nd.array: array [Nx(D+1)]
    """
    return np.hstack((np.ones((X.shape[0], 1)), X))


def augment_features_polynomial(X, M):
    """ # TODO add docstring
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    # TODO: add other types of feature expansions
    if M < 2:
        return X

    index_start = 1 if np.all(X[:, 0] == 1) else 0

    X_poly = X

    for i in range(2, M + 1):
        X_powered = np.power(X[:, index_start:], i)
        X_poly = np.hstack((X_poly, X_powered))

    return X_poly
