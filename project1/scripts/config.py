from implementations import *

DATA_PATH = "../data"

TRAIN_DATA_CSV_PATH = DATA_PATH + "/" + "train.csv"
TEST_DATA_CSV_PATH = DATA_PATH + "/" + "test.csv"

TRAIN_URL = "https://github.com/epfml/ML_course/blob/master/projects/project1/data/train.csv.zip?raw=true"
TEST_URL = "https://github.com/epfml/ML_course/blob/master/projects/project1/data/test.csv.zip?raw=true"

LOG_PATH = "../data/logs"

PREPROCESSED_PATH = DATA_PATH + "/" + "preprocessed"
PREPROCESSED_X_TR_GROUPS_NPY = "../data/preprocessed/X_tr.npy"
PREPROCESSED_Y_TR_GROUPS_NPY = "../data/preprocessed/Y_tr.npy"
PREPROCESSED_GROUP_INDEX_TR_NPY = "../data/preprocessed/group_indx_tr.npy"
PREPROCESSED_X_TE_GROUPS_NPY = "../data/preprocessed/X_te.npy"
PREPROCESSED_Y_TE_GROUPS_NPY = "../data/preprocessed/Y_te.npy"
PREPROCESSED_IDS_TE_GROUPS_NPY = "../data/preprocessed/ids_te.npy"
PREPROCESSED_GROUP_INDEX_TE_NPY = "../data/preprocessed/group_indx_te.npy"

SUBMISSION_PATH = "../data/submissions"

Z_VALUE = 3.0
GRAD_STOP_CONDITION = 1e-15

GROUP_COL_FILTERED_TUPLE = (
    (),  # GROUP 1 (all columns)
    (0,),  # GROUP 2
    (4, 5, 6, 12, 26, 27, 28),  # GROUP 3
    (0, 4, 5, 6, 12, 26, 27, 28),  # GROUP 4
    (4, 5, 6, 12, 23, 24, 25, 26, 27, 28),  # GROUP 5
    (0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28)  # GROUP 6
)

IMPLEMENTATIONS = {
    "Least Squares Gradient Descent": {
        "function": least_squares_GD,
        "max_iters_list": [10],
        "gammas": [0.05],
        "lambdas": [None],
    },
    "Least Squares Stochastic GD": {
        "function": least_squares_SGD,
        "max_iters_list": [10],
        "gammas": [0.1],
        "lambdas": [None],
    },
    "Least Squares using Pseudo-Inverse": {
        "function": least_squares,
        "max_iters_list": [None],
        "gammas": [None],
        "lambdas": [None],
    },
    "Ridge Regression": {
        "function": ridge_regression,
        "max_iters_list": [None],
        "gammas": [None],
        "lambdas": [0.01]
    },
    "Logistic Regression": {
        "function": logistic_regression,
        "max_iters_list": [10],
        "gammas": [0.01],
        "lambdas": [None]
    },
    "Regularized Logistic Regression": {
        "function": reg_logistic_regression,
        "max_iters_list": [10],
        "gammas": [0.01],
        "lambdas": [0.1]
    },
    "Support Vector Machine": {
        "function": support_vector_machine_GD,
        "max_iters_list": [10],
        "gammas": [0.1],
        "lambdas": [0.1]
    },
    "Least Squares Mini-Batch GD": {
        "function": least_squares_BGD,
        "max_iters_list": [10],
        "gammas": [0.5],
        "lambdas": [None]
    },
}
