# -*- coding: utf-8 -*-
"""
This file is responsible for generating a submission.

The generate_best() function of this script is given a dictionary, contain
one entry for each missing value group (see exploratory_data_analysis.ipynb),
with each entry being a dictionary with the best setting. This dictionary can
either be directly supplied to the generate_best() function, or, if not
supplied, the function will look in the /log folder with the name best.json and
will determine the best setting for each group automatically.
"""

import datetime
import json
import sys
import numpy as np
from pathlib import Path

from utils import transform_log_dict_to_param_dict
from proj1_helpers import predict_labels, create_csv_submission
from data_loader import get_data
from config import IMPLEMENTATIONS, SUBMISSION_PATH

START_TIME = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M")
np.random.seed(0)


def generate_submission(ids_te, Y_te):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    # generate submission
    print("[!] Generating Submission...")
    date_time = START_TIME
    # TODO replace whitespaces in function names
    csv_name = f"HB_SUBMISSION_{date_time}.csv"

    Path(SUBMISSION_PATH).mkdir(exist_ok=True)

    create_csv_submission(ids_te, Y_te, csv_name, SUBMISSION_PATH)
    print(f"[+] Submission {csv_name} was generated!")


def generate_best(param_dict=None, log_param_dict_path="../data/logs/best.json"):
    # if not parameters are given manually, look for a log dictionary
    if not param_dict:
        try:
            with open(log_param_dict_path, "r") as f:
                log_dict = json.load(f)
                param_dict = transform_log_dict_to_param_dict(log_dict)
        except OSError:
            print(f"Could not open/read file: {log_param_dict_path}")
            sys.exit()

    M_list = [param_dict[str(group_indx)]["M"] for group_indx in range(1, 7)]
    class_equalizer_list = [param_dict[str(group_indx)]["class_eq"] for group_indx in range(1, 7)]
    z_outlier_list = [param_dict[str(group_indx)]["z_outlier"] for group_indx in range(1, 7)]
    corr_anal_list = [param_dict[str(group_indx)]["corr_anal"] for group_indx in range(1, 7)]

    # divide the dataset into the multiple groups and preprocess it
    # TODO change preexisting to False
    groups_tr_X, groups_tr_Y, indc_list_tr, groups_te_X, groups_te_Y, indc_list_te, ids_te = get_data(
        use_preexisting=False, save_preprocessed=True, z_outlier=z_outlier_list, feature_expansion=True,
        correlation_analysis=corr_anal_list, class_equalizer=class_equalizer_list, M=M_list)
    # numpy array for submission
    Y_te = np.zeros(shape=(568238,))

    # for each group...
    for group_indx, (X_tr, Y_tr, X_te, Y_te_indx) in enumerate(
            zip(groups_tr_X, groups_tr_Y, groups_te_X, indc_list_te)):
        # get shape and create initial parameters
        N, D = X_tr.shape
        W_init = np.random.rand(D, )
        best_params_train = {"tx": X_tr, "y": Y_tr, "initial_w": W_init,
                             "max_iters": param_dict[str(group_indx + 1)]["params"][0],
                             "gamma": param_dict[str(group_indx + 1)]["params"][1],
                             "lambda_": param_dict[str(group_indx + 1)]["params"][2]}

        # train it on all available training data
        W_best, _ = IMPLEMENTATIONS[param_dict[str(group_indx + 1)]["function_name"]]["function"](**best_params_train)

        # write into the corresponding indexes of this group
        Y_te[Y_te_indx] = predict_labels(W_best, X_te)

    generate_submission(ids_te, Y_te)


if __name__ == "__main__":
    best_dictionary = {"1": {
        "M": 5,
        "corr_anal": True,
        "class_eq": False,
        "z_outlier": False,
        "function_name": "Ridge Regression",
        "params": [None, None, 0.8]
    },
        "2": {
            "M": 5,
            "corr_anal": True,
            "class_eq": False,
            "z_outlier": False,
            "function_name": "Ridge Regression",
            "params": [None, None, 0.8]
        },
        "3": {
            "M": 5,
            "class_eq": False,
            "corr_anal": True,
            "z_outlier": False,
            "function_name": "Ridge Regression",
            "params": [None, None, 0.8]
        },
        "4": {
            "M": 5,
            "corr_anal": True,
            "class_eq": False,
            "z_outlier": False,
            "function_name": "Ridge Regression",
            "params": [None, None, 0.8]
        },
        "5": {
            "M": 5,
            "corr_anal": True,
            "class_eq": False,
            "z_outlier": False,
            "function_name": "Ridge Regression",
            "params": [None, None, 0.8]
        },
        "6": {
            "M": 5,
            "corr_anal": True,
            "class_eq": False,
            "z_outlier": False,
            "function_name": "Ridge Regression",
            "params": [None, None, 0.8]
        }
    }
    generate_best()
