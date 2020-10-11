import datetime
import numpy as np
import os.path
from collections import defaultdict

from utils import calculate_mse_loss, cross_validation, build_k_indices
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from preprocessing import z_score_outlier_detection, add_ones_column, \
    augment_features_polynomial, standardize, split_data, corr_filter
from implementations import *

np.random.seed(0)

SUBMISSION_PATH = "../data/submissions"
TRAIN_PATH = "../data/train.csv"

TEST_PATH = "../data/test.csv"
PREPROCESSED_X = "../data/preprocessed_X.npy"
PREPROCESSED_Y = "../data/preprocessed_Y.npy"
PREPROCESSED_X_te = "../data/preprocessed_X_te.npy"
PREPROCESSED_Y_te = "../data/preprocessed_Y_te.npy"
PREPROCESSED_ids_te = "../data/preprocessed_ids_te.npy"

IMPLEMENTATIONS = {
    "Least Squares Gradient Descent": {
        "function": least_squares_GD,
        "max_iters": 100,
        "gamma": 0.01,
        "lambda_": None
    },
    "Least Squares Stochastic GD": {
        "function": least_squares_SGD,
        "max_iters": 100,
        "gamma": 0.01,
        "lambda_": None
    },
    "Least Squares using Pseudo-Inverse": {
        "function": least_squares,
        "max_iters": None,
        "gamma": None,
        "lambda_": None
    },
    "Ridge Regression": {
        "function": ridge_regression,
        "max_iters": 100,
        "gamma": 0.01,
        "lambda_": 0.1
    },
    "Logistic Regression": {
        "function": logistic_regression,
        "max_iters": 500,
        "gamma": 0.01,
        "lambda_": None
    },
    "Regularized Logistic Regression": {
        "function": reg_logistic_regression,
        "max_iters": 500,
        "gamma": 0.1,
        "lambda_": 0.1
    }
}

Z_VALUE = 3.0
DO_Z_OUTLIER_DETECTION = True

K = 5
USE_PRE = False

if __name__ == "__main__":
    if not (os.path.isfile(PREPROCESSED_X) and USE_PRE):
        Y, X, ids = load_csv_data(TRAIN_PATH)
        Y_te, X_te, ids_te = load_csv_data(TEST_PATH)

        # perform preprocessing TODO seems to make it worse
        if DO_Z_OUTLIER_DETECTION:
            X = z_score_outlier_detection(X, thresh=Z_VALUE)
            X_te = z_score_outlier_detection(X_te, thresh=Z_VALUE)

        X, columns_to_keep = corr_filter(X, threshold=0.95)
        X_te = X_te[:, columns_to_keep]

        # Augment feature vector
        # X = augment_features_polynomial(X, M=4)
        # X_te = augment_features_polynomial(X_te, M=4)

        # standardize features
        X = standardize(X)
        X_te = standardize(X_te)

        # add ones
        X = add_ones_column(X)
        X_te = add_ones_column(X_te)

        np.save(PREPROCESSED_X, X, allow_pickle=True)
        np.save(PREPROCESSED_X_te, X_te, allow_pickle=True)
        np.save(PREPROCESSED_Y, Y, allow_pickle=True)
        np.save(PREPROCESSED_Y_te, Y_te, allow_pickle=True)
        np.save(PREPROCESSED_ids_te, ids_te, allow_pickle=True)
        print("[*] Saved Preprocessed Data")
    else:
        print("[*] Using Saved Data")
        X = np.load(PREPROCESSED_X)
        X_te = np.load(PREPROCESSED_X_te)
        Y = np.load(PREPROCESSED_Y)
        Y_te = np.load(PREPROCESSED_Y_te)
        ids_te = np.load(PREPROCESSED_ids_te)

    N, D = X.shape

    # Cross-validation
    k_indices = build_k_indices(Y, K)

    nr_functions = len(IMPLEMENTATIONS.items())
    loss_array_val = np.zeros(shape=(K, nr_functions))
    W_init = np.random.rand(D, )

    for k_iteration in range(K):
        print(f"{k_iteration + 1}. Iteration of cross validation...")
        X_train, Y_train, X_val, Y_val = cross_validation(
            Y, X, k_indices, k_iteration)

        # convert from {-1, 1} to {0, 1} for logistic regression
        Y_log_train = ((Y_train + 1) / 2).astype(int)

        for j, [f_name, f] in enumerate(IMPLEMENTATIONS.items()):
            print(f"[!] Starting {f_name}...")
            W_init = np.random.rand(D, )

            if "Logistic" in f_name:
                Y_f_train = Y_log_train
            else:
                Y_f_train = Y_train

            args_train = {"tx": X_train, "y": Y_f_train, "initial_w": W_init, "max_iters": f["max_iters"],
                          "gamma": f["gamma"], "lambda_": f["lambda_"]}

            W, loss_tr = f["function"](**args_train)

            if "Logistic" in f_name:
                prediction_val = sigmoid(X_val @ W)
            else:
                prediction_val = X_val @ W

            loss_val = calculate_mse_loss(Y_val, prediction_val)
            loss_array_val[k_iteration, j] = loss_val

            print(f"\t [==>] Validation Loss: {loss_val:.2f}")
        print("#########################################################################################")

    loss_array_val = np.mean(loss_array_val, axis=0)

    # get values of best method
    index_best = int(np.argmin(loss_array_val))
    f_best_name = list(IMPLEMENTATIONS.keys())[index_best]
    best = list(IMPLEMENTATIONS.values())[index_best]
    f_best = best["function"]
    loss_val_best = loss_array_val[index_best]
    print(
        f"[+] Best Method was {f_best_name} with a Validation Loss of {loss_val_best}!")

    args_train = {"tx": X, "y": Y, "initial_w": W_init,
                  "max_iters": best["max_iters"],
                  "gamma": best["gamma"],
                  "lambda_": best["lambda_"]}

    W_best, _ = f_best(**args_train)
    # generate submission
    print("[!] Generating Submission...")
    Y_te_predictions = predict_labels(W_best, X_te)

    date_time = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M")
    # TODO replace whitespaces in function names
    csv_name = f"HB_SUBMISSION_{loss_val_best:.2f}_{date_time}_{f_best_name}.csv"
    create_csv_submission(ids_te, Y_te_predictions, csv_name, SUBMISSION_PATH)
    print(f"[+] Submission {csv_name} was generated!")
