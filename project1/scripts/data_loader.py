import numpy as np
import os
import zipfile
import requests

from proj1_helpers import load_csv_data
from preprocessing import standardize, add_bias, augment_features_polynomial, SMOTE, split_groups,\
    z_score_outlier_detection, corr_filter
import config


def download_url(url, save_path, chunk_size=128):
    """
    Function to download a given (zip-)file.

    Args:
        url (string): url to file to be downloaded
        save_path (string): path to save file
        chunk_size (int): size of chunk for download
    """

    print(f"[*] Downloading from [{url}]")
    r = requests.get(url, stream=True)
    with open(save_path + ".zip", 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    print(f"[*] Uncompressing to [{save_path}]")
    with zipfile.ZipFile(f"{save_path}.zip", 'r') as zip_ref:
        zip_ref.extractall(config.DATA_PATH)


def get_data(use_preexisting=True, save_preprocessed=True, z_outlier=False,
             feature_expansion=False, correlation_analysis=False,
             apply_SMOTE=False, M=4):
    """
    Data supplying function.

    This function has the purpose of loading data and applying preprocessing.
    It includes many features such as downloading the data from the github
    repository, saving the data (for fast reuse), applying different
    preprocessing algorithms, etc...

    Args:
        use_preexisting (bool): if existent, enabling this parameters will allow
                                the function to use previously preprocessed and
                                saved data files
        save_preprocessed (bool): enabling this parameters will allow the
                                    function to save the preprocessed data
        z_outlier (bool): enabling this parameters will allow the function to
                            perform z outlier detection
        feature_expansion (bool): enabling this parameters will allow the
                                    function to perform exponential feature
                                    expansion
        correlation_analysis (bool): enabling this parameters will allow the
                                        function to perform correlation analysis
                                        and remove highly correlated features
        apply_SMOTE (bool): enabling this parameters will allow the function to
                            perform SMOTE (Synthetic Minority Oversampling Technique)

    Returns:
        list: groups of training samples
        list: corresponding groups of training labels
        list: corresponding indexes of affiliated training ows
        list: groups of test samples
        list: corresponding groups of test labels
        list: corresponding indexes of affiliated test rows
        list: list of indexes of testing (for creating submissions)

    """

    if os.path.isdir(config.DATA_PATH) and os.path.isdir(config.PREPROCESSED_PATH) and use_preexisting:
        print("[*] Using previously preprocessed Data")
        groups_tr_X = np.load(config.PREPROCESSED_X_TR_GROUPS_NPY, allow_pickle=True)
        groups_tr_Y = np.load(config.PREPROCESSED_Y_TR_GROUPS_NPY, allow_pickle=True)
        indc_list_tr = np.load(config.PREPROCESSED_GROUP_INDEX_TR_NPY, allow_pickle=True)
        groups_te_X = np.load(config.PREPROCESSED_X_TE_GROUPS_NPY, allow_pickle=True)
        groups_te_Y = np.load(config.PREPROCESSED_Y_TE_GROUPS_NPY, allow_pickle=True)
        indc_list_te = np.load(config.PREPROCESSED_GROUP_INDEX_TE_NPY, allow_pickle=True)
        ids_te = np.load(config.PREPROCESSED_IDS_TE_GROUPS_NPY, allow_pickle=True)

    else:
        print("[*] Creating preprocessed Data")

        if not (os.path.isdir(config.DATA_PATH) and os.path.isfile(config.TRAIN_DATA_CSV_PATH) and os.path.isfile(
                config.TEST_DATA_CSV_PATH)):
            if not (os.path.isdir(config.DATA_PATH)):
                os.mkdir(config.DATA_PATH)
            download_url(config.TRAIN_URL, config.TRAIN_DATA_CSV_PATH)
            download_url(config.TEST_URL, config.TEST_DATA_CSV_PATH)

        # load data from csv files
        Y_tr, X_tr, ids_tr = load_csv_data(config.TRAIN_DATA_CSV_PATH)
        Y_te, X_te, ids_te = load_csv_data(config.TEST_DATA_CSV_PATH)

        groups_tr_Y, groups_tr_X, indc_list_tr = split_groups(Y_tr, X_tr)
        groups_te_Y, groups_te_X, indc_list_te = split_groups(Y_te, X_te)

        for indx in range(len(indc_list_tr)):
            # perform z outlier detection
            if z_outlier:
                groups_tr_X[indx] = z_score_outlier_detection(groups_tr_X[indx], thresh=config.Z_VALUE)
                groups_te_X[indx] = z_score_outlier_detection(groups_te_X[indx], thresh=config.Z_VALUE)
            # perform correlation analysis
            if correlation_analysis:
                groups_tr_X[indx], columns_to_keep = corr_filter(groups_tr_X[indx], threshold=0.95)
                groups_te_X[indx] = groups_te_X[indx][:, columns_to_keep]
            # perform SMOTE
            if apply_SMOTE:
                groups_tr_X[indx], groups_tr_Y[indx] = SMOTE(groups_tr_X[indx], groups_tr_Y[indx])

            # perform feature expansion
            if feature_expansion:
                groups_tr_X[indx] = augment_features_polynomial(groups_tr_X[indx], M=M)
                groups_te_X[indx] = augment_features_polynomial(groups_te_X[indx], M=M)

            # standardize features
            groups_tr_X[indx] = standardize(groups_tr_X[indx])
            groups_te_X[indx] = standardize(groups_te_X[indx])

            # add bias
            groups_tr_X[indx] = add_bias(groups_tr_X[indx])
            groups_te_X[indx] = add_bias(groups_te_X[indx])

            print(f"\t [+]Group {indx + 1} finished!")

        if save_preprocessed:
            if not os.path.isdir(config.PREPROCESSED_PATH):
                os.mkdir(config.PREPROCESSED_PATH)

            np.save(config.PREPROCESSED_X_TR_GROUPS_NPY, groups_tr_X, allow_pickle=True)
            np.save(config.PREPROCESSED_Y_TR_GROUPS_NPY, groups_tr_Y, allow_pickle=True)
            np.save(config.PREPROCESSED_X_TE_GROUPS_NPY, groups_te_X, allow_pickle=True)
            np.save(config.PREPROCESSED_Y_TE_GROUPS_NPY, groups_te_Y, allow_pickle=True)
            np.save(config.PREPROCESSED_GROUP_INDEX_TR_NPY, indc_list_tr, allow_pickle=True)
            np.save(config.PREPROCESSED_GROUP_INDEX_TE_NPY, indc_list_te, allow_pickle=True)
            np.save(config.PREPROCESSED_IDS_TE_GROUPS_NPY, ids_te, allow_pickle=True)
            print("[+] Saved Preprocessed Data")

    return groups_tr_X, groups_tr_Y, indc_list_tr, groups_te_X, groups_te_Y, indc_list_te, ids_te


if __name__ == "__main__":
    a, b, c, d, e, f, g = get_data()
