import datetime
import numpy as np
np.random.seed(0)
import os.path
from collections import defaultdict

# TODO remove functions that are not used here
from utils import calculate_mse_loss, calculate_acc, cross_validation, build_k_indices
from proj1_helpers import predict_labels, create_csv_submission
from data_loader import get_data
from implementations import *
from global_variables import *

if __name__ == "__main__": # COMMENT CODE BELOW TODO
    K = 5

    groups_tr_X, groups_tr_Y, indc_list_tr, groups_te_X, groups_te_Y, indc_list_te, ids_te = get_data(use_preexisting=True,
        save_preprocessed=True, z_outlier=False, feature_expansion=False, correlation_analysis=False)

    Y_te = np.zeros((568238))
    for group_indx, (X_tr, Y_tr, X_te, Y_te_indx) in enumerate(zip(groups_tr_X, groups_tr_Y, groups_te_X, indc_list_te)):
        print("="*240)
        print("GROUP {}".format(group_indx+1))

        N, D = X_tr.shape

        # Cross-validation
        k_indices = build_k_indices(Y_tr, K)

        nr_functions = len(IMPLEMENTATIONS.items())
        loss_array_val = np.zeros(shape=(K, nr_functions))
        W_init = np.random.rand(D, )

        for k_iteration in range(K):
            print(f"{k_iteration + 1}. Iteration of cross validation...")
            X_train, Y_train, X_val, Y_val = cross_validation(
                Y_tr, X_tr, k_indices, k_iteration)

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
                acc_val = calculate_acc(Y_val, prediction_val)
                print(f"\t [==>] Validation Loss: {loss_val:.2f} | Validation Accuracy: {acc_val:.2f}")
            print("################################################################################")

        loss_array_val = np.mean(loss_array_val, axis=0)

        # get values of best method
        index_best = int(np.argmin(loss_array_val))
        f_best_name = list(IMPLEMENTATIONS.keys())[index_best]
        best = list(IMPLEMENTATIONS.values())[index_best]
        f_best = best["function"]
        loss_val_best = loss_array_val[index_best]
        print(
            f"[+] Best Method was {f_best_name} with a Validation Loss of {loss_val_best}!")

        args_train = {"tx": X_tr, "y": Y_tr, "initial_w": W_init,
                      "max_iters": best["max_iters"],
                      "gamma": best["gamma"],
                      "lambda_": best["lambda_"]}

        W_best, _ = f_best(**args_train)


        Y_te[Y_te_indx] = predict_labels(W_best, X_te)

    # generate submission
    print("[!] Generating Submission...")
    date_time = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M")
    # TODO replace whitespaces in function names
    csv_name = f"HB_SUBMISSION_{date_time}.csv"
    if not (os.path.isdir(SUBMISSION_PATH)):
        os.mkdir(SUBMISSION_PATH)
    create_csv_submission(ids_te, Y_te, csv_name, SUBMISSION_PATH)
    print(f"[+] Submission {csv_name} was generated!")
