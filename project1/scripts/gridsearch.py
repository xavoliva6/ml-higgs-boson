import datetime
import itertools
import json
import numpy as np
from pathlib import Path

from utils import calculate_mse_loss, calculate_acc, cross_validation_iter, build_k_indices, sigmoid, create_labels
from data_loader import get_data
from config import IMPLEMENTATIONS, LOG_PATH

START_TIME = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M")
np.random.seed(0)


def cross_validation(k, X, y, params, regression):
    """
    Performing regression using K-Cross Validation.

    This function is used to generate a model, given data, a regression function
    and a set of parameters.

    Args:
        k (int): k for cross validation
        X (nd.array): training samples of form N x D
        y (nd.array): training samples of form N
        params (dict): dictionary of training samples
        regression (function): regression function

    Returns:
        float: mean loss on validation datasets
        float: mean accuracy on validation datasets

    Raise:
        ValueError: if the regression function raises an error
    """

    # Cross-validation
    k_indices = build_k_indices(y, k)
    accuracies = []
    losses = []

    # print(f"(max_iters: {params['max_iters']}, gamma: {params['gamma']}, lambda: {params['lambda_']})")
    # each iteration for each split of training and validation
    for k_iteration in range(k):
        # split the data accordingly into training and validation
        X_train, Y_train, X_val, Y_val = cross_validation_iter(y, X, k_indices, k_iteration)
        # initial weights
        W_init = np.random.rand(D, )
        # initialize dictionary for the training regression model
        args_train = {"tx": X_train, "y": Y_train, "initial_w": W_init, "max_iters": params["max_iters"],
                      "gamma": params["gamma"], "lambda_": params["lambda_"]}
        # try to train the model, if this doesnt work, raise an error
        try:
            W, loss_tr = regression(**args_train)
        except ValueError:
            print("Regression diverged with these parameters.")
            return None, None
        # TODO
        if "Logistic" in f_name:
            prediction_val_regression = sigmoid(X_val @ W)
        else:
            prediction_val_regression = X_val @ W
        # calculate prediction for the validation dataset
        prediction_val = create_labels(prediction_val_regression)
        # calculate corresponding loss and accuracy
        loss_val = calculate_mse_loss(Y_val, prediction_val)
        acc_val = calculate_acc(Y_val, prediction_val)
        losses.append(loss_val)
        accuracies.append(acc_val)
    # finally, generate the means
    mean_loss_val = np.array(losses).mean()
    mean_acc_val = np.array(accuracies).mean()
    # print(kkk*4 + f"\t [==>] Val_Loss: {mean_loss_val:.2f} | Val_Acc: {mean_acc_val:.2f}")

    return mean_loss_val, mean_acc_val


if __name__ == "__main__":
    M = [30, 5]
    z_outlier = [True, False]
    correlation_analysis = [True, False]
    class_equalizer = [True, False]
    K = 2
    kkk = "---- "

    # create log folder
    Path(LOG_PATH).mkdir(exist_ok=True)
    # define the name of the log file
    log_file_name = "log_gridsearch_" + START_TIME + ".json"
    # log data is a dictionary, with groups as indexes
    log_dict = {group_indx: [] for group_indx in range(1, 7)}

    for m in M:
        for z_outlier_bool in z_outlier:
            for correlation_analysis_bool in correlation_analysis:
                for class_equalizer_bool in class_equalizer:
                    print("=" * 80)
                    print(
                        f" Preprocess Setup: M:{m} | ZOD:{z_outlier_bool} | CA:{correlation_analysis_bool} | CE:{class_equalizer_bool}")
                    # divide the dataset into the multiple groups and preprocess them
                    groups_tr_X, groups_tr_Y, indc_list_tr, groups_te_X, groups_te_Y, indc_list_te, ids_te = get_data(
                        use_preexisting=False, save_preprocessed=False, z_outlier=z_outlier_bool,
                        feature_expansion=True,
                        correlation_analysis=correlation_analysis_bool, class_equalizer=class_equalizer_bool, M=m)

                    # for each group...
                    for group_indx, (X_tr, Y_tr, X_te, Y_te_indx) in enumerate(
                            zip(groups_tr_X, groups_tr_Y, groups_te_X, indc_list_te), start=1):
                        # print("=" * 240)
                        print(kkk * 2 + f"Group: {group_indx}")

                        # get the shape of the sample array
                        N, D = X_tr.shape

                        # initialize parameters z_outlier_bool
                        index_best_total = 0
                        acc_best_total = 0

                        # go through each function
                        for j, [f_name, f] in enumerate(IMPLEMENTATIONS.items()):
                            print(kkk * 3 + f"Function: {f_name}...")
                            # create grid for grid search
                            grid = itertools.product(f["max_iters_list"], f["gammas"], f["lambdas"])
                            nr_configs = len(f["max_iters_list"]) * len(f["gammas"]) * len(f["lambdas"])
                            # array for saving accuracy
                            acc_array_val = np.zeros(shape=(nr_configs))
                            # for each parameter setup
                            for i, params in enumerate(grid):
                                params_dict = {
                                    "max_iters": params[0],
                                    "gamma": params[1],
                                    "lambda_": params[2]
                                }
                                # calculate the loss and accuracy
                                mean_loss, mean_acc = cross_validation(K, X_tr, Y_tr, params_dict, f["function"])
                                acc_array_val[i] = mean_acc

                            # get values of best method of this regressionrun
                            index_best = int(np.argmax(acc_array_val))
                            acc_val_best = acc_array_val[index_best]
                            # print(kkk*4 +
                            #       f"[+] Best {f_name} with an accuracy of {acc_val_best} - max_iters: {params[0]}, gamma: {params[1]}, lambda: {params[2]}")
                            if acc_val_best > acc_best_total:
                                acc_best_total = acc_val_best
                                index_best_total = (j, index_best)

                        # get best model
                        for j, [f_name, f] in enumerate(IMPLEMENTATIONS.items()):
                            if j == index_best_total[0]:
                                f_best_name = f_name
                                f_best = f

                        # and the best parameters
                        grid = itertools.product(f_best["max_iters_list"], f_best["gammas"], f_best["lambdas"])
                        for i, params in enumerate(grid):
                            if i == index_best_total[1]:
                                best_params = params

                        # each tuple (acc, {parameters/function}) will be
                        # appended to the list of the corresponding group
                        with open(LOG_PATH + "/" + log_file_name, "w") as f:
                            log_dict[group_indx].append((acc_best_total,
                                                         {"function": f_best_name,
                                                          "params": best_params,
                                                          "M": m,
                                                          "Z": z_outlier_bool,
                                                          "CA": correlation_analysis_bool,
                                                          "CE": class_equalizer_bool
                                                          }))
                            json.dump(log_dict, f, indent=4)
