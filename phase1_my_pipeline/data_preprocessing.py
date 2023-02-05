"""Data preprocessing methods.

Authors:
    Keer Ni - knni@ucdavis.edu

"""

import copy
import sys
import warnings
import numpy as np
import sklearn.neighbors._base

from sklearn.impute import KNNImputer
from missingpy import MissForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base # noqa



"""Missing value imputation methods.

1. KNN
2. missForest

"""
# hyperparameter tuning for missing value imputation --cross validation?

def KNN_impute(re_train_data):
    """Impute missing data using KNN.

    Args:
        re_train_data (np array): train data with random removal.

    Returns:
        im_train_data (np array): imputed train data.

    """
    X = copy.deepcopy(re_train_data)
    nan = np.nan

    # make np.nan when creating the data directly
    for i in range(len(X)):
        for j in range(len(X[0])):
            if X[i][j] == None:
                X[i][j] = nan

    imputer = KNNImputer(n_neighbors = 5, weights="uniform")
    im_train_data = imputer.fit_transform(X)

    return im_train_data


def MissForest_impute(re_train_data):
    """Impute missing data using MissForest.

    Args:
        re_train_data (np array): train data with random removal.

    Returns:
        im_train_data (np array): imputed train data.

    """
    X = copy.deepcopy(re_train_data)
    nan = float("NaN")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i in range(len(X)):
            for j in range(len(X[0])):
                if X[i][j] == None:
                    X[i][j] = nan

        imputer = MissForest()
        im_train_data = imputer.fit_transform(X)

    return im_train_data



"""Feature scaling methods.

1. MinMax
2. Standardization

"""

def MinMax_scale(im_train_data):
    """Scale data using MinMax.

    Args:
        im_train_data (np array): imputed train data.

    Returns:
        sc_train_data (np array): scaled train data.

    """
    scaler = MinMaxScaler()

    sc_train_data = scaler.fit(im_train_data).transform(im_train_data)

    return sc_train_data


def Standardize_scale(im_train_data):
    """Scale data using Standardization.

    Args:
        im_train_data (np array): imputed train data.

    Returns:
        sc_train_data (np array): scaled train data.

    """
    scaler = StandardScaler()

    sc_train_data = scaler.fit(im_train_data).transform(im_train_data)

    return sc_train_data



"""Outlier detection methods.

1. LOF
2. IsolationForest

"""

def LOF_outlier(im_train_data):
    """Detect outlier using LOF.

    Args:
        im_train_data (np array): imputed train data.

    Returns:
        (list): indices of detected outliers.

    """
    clf = LocalOutlierFactor(n_neighbors=2)

    is_outlier = list(clf.fit_predict(im_train_data))

    return [i for i, x in enumerate(is_outlier) if x != 1]


def IsolationForest_outlier(im_train_data):
    """Detect outlier using IsolationForest.

    Args:
        im_train_data (np array): imputed train data.

    Returns:
        (list): indices of detected outliers.

    """
    clf = IsolationForest(random_state=0).fit(im_train_data)

    is_outlier = list(clf.predict(im_train_data))

    return [i for i, x in enumerate(is_outlier) if x != 1]

# PCA
# TSNE // generally better

if __name__ == '__main__':
    """
        Using the iris dataset from scikit learn.
        Exporting npremoved_train_data, nporginal_train_data, nptrain_data_labels, nptest_data_labels and print to see if they are the same.
    """

    with open('create_test_data.npy', 'rb') as f:
        npremoved_train_data = np.load(f)
        nporiginal_train_data = np.load(f)
        nptrain_data_labels = np.load(f)
        nptest_data = np.load(f)
        nptest_data_labels = np.load(f)
    print(npremoved_train_data)
    print(nporiginal_train_data)
    print(nptrain_data_labels)
    print(nptest_data)
    print(nptest_data_labels)
