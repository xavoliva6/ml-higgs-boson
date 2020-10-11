import datetime
import numpy as np
import os.path

from utils import split_data, standardize, calculate_mse_loss, cross_validation, build_k_indices
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from preprocessing import z_score_outlier_detection, add_ones_column, augment_features_polynomial
from implementations import *

SUBMISSION_PATH = "../data/submissions"
TRAIN_PATH = "../data/train.csv"

TEST_PATH = "../data/test.csv"
PREPROCESSED_X = "../data/preprocessed_X.npy"
PREPROCESSED_Y = "../data/preprocessed_Y.npy"
PREPROCESSED_X_te = "../data/preprocessed_X_te.npy"
PREPROCESSED_Y_te = "../data/preprocessed_Y_te.npy"


IMPLEMENTATIONS = {"Least Squares Gradient Descent": least_squares_GD,
                   "Least Squares Stochastic GD": least_squares_SGD,
                   "Least Squares using Pseudo-Inverse": least_squares,
                   "Ridge Regression": ridge_regression,
                   "Logistic Regression": logistic_regression,
                   "Regularized Logistic Regression": reg_logistic_regression}

Z_VALUE = 3.0
DO_Z_OUTLIER_DETECTION = True

MAX_ITERS = 100
GAMMA = .01
LAMBDA_ = .1
K = 5
USE_PRE = True
if __name__ == "__main__":
    print(os.path.isfile(PREPROCESSED_X))
    if not (os.path.isfile(PREPROCESSED_X) and USE_PRE):
        Y, X, ids = load_csv_data(TRAIN_PATH)
        Y_te, X_te, ids_te = load_csv_data(TEST_PATH)

        # perform preprocessing TODO seems to make it worse
        if DO_Z_OUTLIER_DETECTION:
            X = z_score_outlier_detection(X, thresh=Z_VALUE)
            X_te = z_score_outlier_detection(X_te, thresh=Z_VALUE)

        # Augment feature vector
        X = augment_features_polynomial(X, M=4)
        X_te = augment_features_polynomial(X_te, M=4)

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
        print("[*] Saved Preprocessed Data")
    else:
        print("[*] Using Saved Data")
        X = np.load(PREPROCESSED_X)
        X_te = np.load(PREPROCESSED_X_te)
        Y = np.load(PREPROCESSED_Y)
        Y_te = np.load(PREPROCESSED_Y_te)

    N, D = X.shape

    # Cross-validation
    k_indices = build_k_indices(Y, K)

    nr_functions = len(IMPLEMENTATIONS.items())
    loss_array_val = np.zeros(shape=(K, nr_functions))
    W_init = np.random.rand(D, )

    for k_iteration in range(K):
        print(f"{k_iteration + 1}. Iteration of cross validation...\n")
        X_train, Y_train, X_val, Y_val = cross_validation(Y, X, k_indices, k_iteration)

        for j, [f_name, f] in enumerate(IMPLEMENTATIONS.items()):
            print("[!] Starting {}...".format(f_name))

            args_train = {"tx": X_train, "y": Y_train, "initial_w": W_init, "max_iters": MAX_ITERS,
                          "gamma": GAMMA, "lambda_": LAMBDA_}

            W, loss_tr = f(**args_train)

            prediction_val = X_val @ W

            loss_val = calculate_mse_loss(Y_val, prediction_val)
            print(loss_val)
            loss_array_val[k_iteration, j] = loss_val

            print(f"\t [==>] Validation Loss: {loss_val}")
        print("#########################################################################################\n")

    loss_array_val = np.mean(loss_array_val, axis=0)

    # get values of best method
    index_best = int(np.argmin(loss_array_val))
    f_best_name = list(IMPLEMENTATIONS.keys())[index_best]
    f_best = list(IMPLEMENTATIONS.values())[index_best]
    loss_val_best = loss_array_val[index_best]
    print(f"[+] Best Method was {f_best_name} with a Validation Loss of {loss_val_best}!")

    args_train = {"tx": X, "y": Y, "initial_w": W_init, "max_iters": MAX_ITERS,
                  "gamma": GAMMA, "lambda_": LAMBDA_}

    W_best, _ = f_best(**args_train)
    # generate submission
    print("[!] Generating Submission...")
    Y_te_predictions = predict_labels(W_best, X_te)

    date_time = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M")
    # TODO replace whitespaces in function names
    csv_name = f"HB_SUBMISSION_{loss_val_best}_{date_time}_{f_best_name}.csv"
    create_csv_submission(ids_te, Y_te_predictions, csv_name, SUBMISSION_PATH)
    print(f"[+] Submission {csv_name} was generated!")
