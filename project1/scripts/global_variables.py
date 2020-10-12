from implementations import *

DATA_PATH = "../data"

TRAIN_DATA_CSV_PATH = DATA_PATH + "/" + "train.csv"
TEST_DATA_CSV_PATH = DATA_PATH + "/" + "test.csv"

TRAIN_URL = "https://github.com/epfml/ML_course/blob/master/projects/project1/data/train.csv.zip?raw=true"
TEST_URL = "https://github.com/epfml/ML_course/blob/master/projects/project1/data/test.csv.zip?raw=true"

PREPROCESSED_PATH = DATA_PATH + "/" + "preprocessed"
PREPROCESSED_X_TR_GROUPS_NPY = "../data/preprocessed/X_tr.npy"
PREPROCESSED_Y_TR_GROUPS_NPY = "../data/preprocessed/Y_tr.npy"
PREPROCESSED_GROUP_INDX_TR_NPY = "../data/preprocessed/group_indx_tr.npy"
PREPROCESSED_X_TE_GROUPS_NPY = "../data/preprocessed/X_te.npy"
PREPROCESSED_Y_TE_GROUPS_NPY = "../data/preprocessed/Y_te.npy"
PREPROCESSED_IDS_TE_GROUPS_NPY = "../data/preprocessed/ids_te.npy"
PREPROCESSED_GROUP_INDX_TE_NPY = "../data/preprocessed/group_indx_te.npy"

SUBMISSION_PATH = "../data/submissions"

IMPLEMENTATIONS = {
    "Least Squares Gradient Descent": {
        "function": least_squares_GD,
        "max_iters": 100,
        "gamma": 0.005,
        "lambda_": None
    },
    "Least Squares Stochastic GD": {
        "function": least_squares_SGD,
        "max_iters": 100,
        "gamma": 0.005,
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
        "max_iters": 100,
        "gamma": 0.01,
        "lambda_": None
    },
    "Regularized Logistic Regression": {
        "function": reg_logistic_regression,
        "max_iters": 100,
        "gamma": 0.1,
        "lambda_": 0.1
    }
}
