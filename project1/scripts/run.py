import datetime
import numpy as np

from utils import split_data, standardize, calculate_mse_loss, cross_validation, build_k_indices
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from preprocessing import z_score_outlier_detection, add_ones_column, augment_features_polynomial
from implementations import *

SUBMISSION_PATH = "submissions"
TRAIN_PATH = "../data/train.csv"
TEST_PATH = "../data/test.csv"
IMPLEMENTATIONS = {"Least Squares Gradient Descent": least_squares_GD,
                   "Least Squares Stochastic GD": least_squares_SGD,
                   "Least Squares using Pseudo-Inverse": least_squares,
                   "Ridge Regression": ridge_regression,
                   "Logistic Regression": logistic_regression,
                   "Regularized Logistic Regression": reg_logistic_regression}

PREPROCESSED_DATA_PATH_TRAIN = "../data/preprocessed_X.npy"
PREPROCESSED_DATA_PATH_TEST = "../data/preprocessed_X_test.npy"
PERFORM_PREPROCESSING = True

Z_VALUE = 3.0
DO_Z_OUTLIER_DETECTION = False

MAX_ITERS = 100
GAMMA = .01
LAMBDA_ = .1

if __name__ == "__main__":
    Y, X, ids = load_csv_data(TRAIN_PATH)
    Y_te, X_te, ids_te = load_csv_data(TEST_PATH)

    if PERFORM_PREPROCESSING:
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

        np.save(PREPROCESSED_DATA_PATH_TRAIN, X)
        np.save(PREPROCESSED_DATA_PATH_TEST, X_te)
    else:
        X = np.load(PREPROCESSED_DATA_PATH_TRAIN)
        X_te = np.load(PREPROCESSED_DATA_PATH_TEST)

    N, D = X.shape

    # Cross-validation
    K = 5
    k_indices = build_k_indices(Y, K)

    nr_functions = len(IMPLEMENTATIONS.items())
    loss_array_val = np.zeros(shape=(K, nr_functions))
    W_init = np.random.rand(D, )

    for k_iteration in range(K):
        print(f"{k_iteration + 1}. Iteration of cross validation...\n")
        X_train, Y_train, X_val, Y_val = cross_validation(Y, X, k_indices, k_iteration)

        for j, [f_name, f] in enumerate(IMPLEMENTATIONS.items()):
            print(f"[!] Starting {f_name}...")

            args_train = {"tx": X_train, "y": Y_train, "initial_w": W_init, "max_iters": MAX_ITERS,
                          "gamma": GAMMA, "lambda_": LAMBDA_}

            W, loss_tr = f(**args_train)

            prediction_val = X_val @ W

            loss_val = calculate_mse_loss(Y_val, prediction_val)
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
