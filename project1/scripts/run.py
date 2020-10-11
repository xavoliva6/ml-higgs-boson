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

Z_VALUE = 3.0
DO_Z_OUTLIER_DETECTION = False

MAX_ITERS = 100
GAMMA = .01
LAMBDA_ = .1

if __name__ == "__main__":
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

    N, D = X.shape

    # Cross-validation
    K = 5
    k_indices = build_k_indices(Y, K)

    nr_functions = len(IMPLEMENTATIONS.items())
    loss_array_val = np.zeros(shape=(K, nr_functions))
    W_init = np.random.rand(D, )
    # W_array = np.zeros(shape=(K, nr_functions, D))

    for k_iteration in range(K):
        print(f"{k_iteration}. Iteration of cross validation...\n")
        X_train, Y_train, X_val, Y_val = cross_validation(Y, X, k_indices, k_iteration)

        for j, [f_name, f] in enumerate(IMPLEMENTATIONS.items()):
            print(f"[!] Starting {f_name}...")

            args_train = {"tx": X_train, "y": Y_train, "initial_w": W_init, "max_iters": MAX_ITERS,
                          "gamma": GAMMA, "lambda_": LAMBDA_}

            W, loss_tr = f(**args_train)
            # W_array[k_iteration, j] = W

            prediction_val = X_val @ W

            loss_val = calculate_mse_loss(Y_val, prediction_val)
            loss_array_val[k_iteration, j] = loss_val

            print(f"\t [==>] Validation Loss: {loss_val}")

    loss_array_val = np.mean(loss_array_val, axis=0)

    # get values of best method
    indx_best = np.argmin(loss_array_val)
    f_best_name = list(IMPLEMENTATIONS.keys())[indx_best]
    f_best = list(IMPLEMENTATIONS.values())[indx_best]
    loss_val_best = loss_array_val[indx_best]
    print(f"[+] Best Method was {f_best_name} with a Validation Loss of {loss_val_best}!")

    args_train = {"tx": X, "y": Y, "initial_w": W_init, "max_iters": MAX_ITERS,
                  "gamma": GAMMA, "lambda_": LAMBDA_}

    W_best, _ = f_best(**args_train)
    # generate submission
    print("[!] Generating Submission...")
    Y_te_predictions = predict_labels(W_best, X_te)

    date_time = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    # TODO replace whitespaces in function names
    csv_name = f"HB_SUBMISSION_{loss_val_best}_{date_time}_{f_best_name}.CSV"
    create_csv_submission(ids_te, Y_te_predictions, csv_name, SUBMISSION_PATH)
    print(f"[+] Submission {csv_name} was generated!")
