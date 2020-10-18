import datetime

import numpy as np
import os.path
import itertools

from utils import calculate_mse_loss, calculate_acc, cross_validation_iter, build_k_indices, sigmoid, create_labels
from proj1_helpers import predict_labels, create_csv_submission
from data_loader import get_data
from config import IMPLEMENTATIONS, SUBMISSION_PATH

np.random.seed(0)


def cross_validation(k, X, y, params, regression):
    # Cross-validation
    k_indices = build_k_indices(y, k)
    accuracies = []
    losses = []
    print(f"(max_iters: {params['max_iters']}, gamma: {params['gamma']}, lambda: {params['lambda_']})")
    for k_iteration in range(k):
        X_train, Y_train, X_val, Y_val = cross_validation_iter(y, X, k_indices, k_iteration)

        W_init = np.random.rand(D, )

        args_train = {"tx": X_train, "y": Y_train, "initial_w": W_init, "max_iters": params["max_iters"],
                      "gamma": params["gamma"], "lambda_": params["lambda_"]}

        try:
            W, loss_tr = regression(**args_train)
        except ValueError:
            print("Regression diverged with these parameters.")
            return None, None

        if "Logistic" in f_name:
            prediction_val_regression = sigmoid(X_val @ W)
        else:
            prediction_val_regression = X_val @ W

        prediction_val = create_labels(prediction_val_regression)

        loss_val = calculate_mse_loss(Y_val, prediction_val)
        acc_val = calculate_acc(Y_val, prediction_val)
        losses.append(loss_val)
        accuracies.append(acc_val)

    mean_loss_val = np.array(losses).mean()
    mean_acc_val = np.array(accuracies).mean()
    print(
        f"\t [==>] Validation Loss: {mean_loss_val:.2f} | Validation Accuracy: {mean_acc_val:.2f}")

    return mean_loss_val, mean_acc_val


def generate_submission(ids_te, Y_te):
    # generate submission
    print("[!] Generating Submission...")
    date_time = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M")
    # TODO replace whitespaces in function names
    csv_name = f"HB_SUBMISSION_{date_time}.csv"
    if not (os.path.isdir(SUBMISSION_PATH)):
        os.mkdir(SUBMISSION_PATH)
    create_csv_submission(ids_te, Y_te, csv_name, SUBMISSION_PATH)
    print(f"[+] Submission {csv_name} was generated!")


if __name__ == "__main__":  # COMMENT CODE BELOW TODO
    groups_tr_X, groups_tr_Y, indc_list_tr, groups_te_X, groups_te_Y, indc_list_te, ids_te = get_data(
        use_preexisting=True, save_preprocessed=True, z_outlier=False, feature_expansion=True,
        correlation_analysis=False, class_equalizer=False, M=4)
    K = 5
    Y_te = np.zeros(shape=(568238,))
    for group_indx, (X_tr, Y_tr, X_te, Y_te_indx) in enumerate(
            zip(groups_tr_X, groups_tr_Y, groups_te_X, indc_list_te)):
        print("=" * 240)
        print(f"GROUP {group_indx + 1}")
        N, D = X_tr.shape
        index_best_total = 0
        acc_best_total = 0
        for j, [f_name, f] in enumerate(IMPLEMENTATIONS.items()):
            print(f"[!] Starting {f_name}...")
            grid = itertools.product(f["max_iters_list"], f["gammas"], f["lambdas"])
            nr_configs = len(f["max_iters_list"]) * len(f["gammas"]) * len(f["lambdas"])
            acc_array_val = np.zeros(shape=(nr_configs))
            for i, params in enumerate(grid):
                params_dict = {
                    "max_iters": params[0],
                    "gamma": params[1],
                    "lambda_": params[2]
                }
                mean_loss, mean_acc = cross_validation(K, X_tr, Y_tr, params_dict, f["function"])
                acc_array_val[i] = mean_acc

            # get values of best method of this regression
            index_best = int(np.argmax(acc_array_val))
            acc_val_best = acc_array_val[index_best]
            print(
                f"[+] Best {f_name} with an accuracy of {acc_val_best} - max_iters: {params[0]}, gamma: {params[1]}, lambda: {params[2]}")
            if acc_val_best > acc_best_total:
                index_best_total = (j, index_best)

        W_init = np.random.rand(D, )
        # Train best model
        for j, [f_name, f] in enumerate(IMPLEMENTATIONS.items()):
            if j == index_best_total[0]:
                f_best_name = f_name
                f_best = f

        grid = itertools.product(f_best["max_iters_list"], f_best["gammas"], f_best["lambdas"])
        for i, params in enumerate(grid):
            if i == index_best_total[1]:
                best_params = params

        best_params_train = {"tx": X_tr, "y": Y_tr, "initial_w": W_init,
                             "max_iters": best_params[0],
                             "gamma": best_params[1],
                             "lambda_": best_params[2]}
        W_best, _ = f_best["function"](**best_params_train)

        Y_te[Y_te_indx] = predict_labels(W_best, X_te)

    generate_submission(ids_te, Y_te)
