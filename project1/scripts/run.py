import datetime
import numpy as np

from utils import split_data, standardize, calculate_mse_loss
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from preprocessing import z_score_outlier_detection
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
VAL_PERCENTAGE = 0.4
Z_VALUE = 3.0
DO_Z_OUTLIER_DETECTION = False

if __name__ == "__main__":
    Y, X, ids = load_csv_data(TRAIN_PATH)
    Y_te, X_te, ids_te = load_csv_data(TEST_PATH)

    N, D = X.shape

    # perform preprocessing TODO seems to make it worse
    if DO_Z_OUTLIER_DETECTION:
        X = z_score_outlier_detection(X, thresh=Z_VALUE)
        X_te = z_score_outlier_detection(X_te, thresh=Z_VALUE)

    # standardize features TODO B4 or after split?
    X = standardize(X)
    X_te = standardize(X_te)

    # add ones
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    X_te = np.hstack((np.ones((X_te.shape[0], 1)), X_te))

    # split data
    (X_tr, Y_tr, ids_tr), (X_val, Y_val, ids_val) = split_data(X, Y, ids, val_prop=VAL_PERCENTAGE)

    max_iters, gamma, lambda_ = 100, .01, .1

    loss_list_val = []
    W_list = []

    for f_name, f in IMPLEMENTATIONS.items():
        print("[!] Starting {} ...".format(f_name))

        W_init = np.random.rand((D + 1), )
        args_train = {"tx": X_tr, "y": Y_tr, "initial_w": W_init, "max_iters": max_iters,
                      "gamma": gamma, "lambda_": lambda_}

        W, loss_tr = f(**args_train)
        W_list.append(W)

        prediction_val = X_val @ W

        loss_val = calculate_mse_loss(Y_val, prediction_val)
        loss_list_val.append(loss_val)

        print("\t [==>] Validation Loss: {}".format(loss_val))

    # get values of best method
    indx_best = np.argmin(loss_list_val)
    f_best_name = list(IMPLEMENTATIONS.keys())[indx_best]
    f_best = list(IMPLEMENTATIONS.values())[indx_best]
    loss_val_best = loss_list_val[indx_best]
    print("[+] Best Method was ({}) with a Validation Loss of {}!".format(
        f_best_name, loss_val_best))

    args_train = {"tx": X, "y": Y, "initial_w": W_init, "max_iters": max_iters,
                "gamma": gamma, "lambda_": lambda_}

    W_best, _ = f_best(**args_train)
    # generate submission
    print("[!] Generating Submission...")
    Y_te_predictions = predict_labels(W_best, X_te)

    date_time = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    # TODO replace whitespaces in function names
    csv_name = "HB_SUBMISSION_{}_{}_{}.CSV".format(loss_val_best, date_time, f_best_name)
    create_csv_submission(ids_te, Y_te_predictions, csv_name, SUBMISSION_PATH)
    print("[+] Submission ({}) was generated!".format(csv_name))
