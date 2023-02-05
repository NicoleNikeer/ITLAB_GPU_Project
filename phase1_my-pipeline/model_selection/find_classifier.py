"""Finding the best classifier.
    1) implement the classifier itself with pipeline
    2) implement the classifier with GridSearch

Authors:
    Keer Ni - knni@ucdavis.edu

"""

"""Perform classification.

1. SVC
2. RandomForest
3. MLP

"""

# get the target labels

import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split



def SVC_classifier(preprocessed_train_data, train_data_labels):
    """Train SVC classifier using given dataset and specific parameters.

    Args:
        preprocessed_train_data (np array): preprocessed train data.
        train_data_labels (np array): train data labels.

    Returns:
        (sklearn.pipeline.Pipeline): object/classifier after using SVC with specific parameters.

    """
    # using the SVC classifier
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=0.25))
    # return the selected parameters for this estimator
    return clf.fit(preprocessed_train_data, train_data_labels)

def SVC_classifier_GridSearchCV(preprocessed_train_data, train_data_labels):
    """Train SVC classifier using given dataset and select best model with optimal parameters.

    Args:
        preprocessed_train_data (np array): preprocessed train data.
        train_data_labels (np array): train data labels.

    Returns:
        (dictionary): dictionary object contians information relate to different parameter combinations.

    """
    # setting search range for parameters
    parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), # 'precomputed' need dataset to be square matrix
                  'C':[1, 10],
                  'gamma':('scale', 'auto')}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    # return the selected parameters for this estimator
    return clf.fit(preprocessed_train_data, train_data_labels).cv_results_



def RandomForest_classifier(preprocessed_train_data, train_data_labels):
    """Train RandomForest classifier using given dataset and select best model with optimal parameters.

    Args:
        preprocessed_train_data (np array): preprocessed train data.
        train_data_labels (np array): train data labels.

    Returns:
        (sklearn.pipeline.Pipeline): object/classifier after using RandomForest with specific parameters.

    """
    # using the RandomForest classifier
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=64, random_state=0, n_estimators=70))
    # return the selected parameters for this estimator
    return clf.fit(preprocessed_train_data, train_data_labels)

def RandomForest_classifier_GridSearchCV(preprocessed_train_data, train_data_labels):
    """Train RandomForest classifier using given dataset and select best model with optimal parameters.

    Args:
        preprocessed_train_data (np array): preprocessed train data.
        train_data_labels (np array): train data labels.

    Returns:
        (dictionary): dictionary object contians information relate to different parameter combinations.

    """
    # setting search range for parameters
    parameters = {'n_estimators':list(range(30, 71, 5)),
                  'criterion':('gini', 'entropy', 'log_loss'),
                  'max_depth':[2, 4, 8, 16, 32, 64]}
    rfc = RandomForestClassifier()
    clf = GridSearchCV(rfc, parameters)
    # return the selected parameters for this estimator
    return clf.fit(preprocessed_train_data, train_data_labels).cv_results_



def MLP_classifier(preprocessed_train_data, train_data_labels):
    """Train MLP classifier using given dataset and select best model with optimal parameters.

    Args:
        preprocessed_train_data (np array): preprocessed train data.
        train_data_labels (np array): train data labels.

    Returns:
        (sklearn.pipeline.Pipeline): object/classifier after using MLP with specific parameters.

    """
    # using the MLP classifier
    clf = MLPClassifier(random_state=1, max_iter=300).fit(preprocessed_train_data, train_data_labels)
    # return the selected parameters for this estimator
    return clf.fit(preprocessed_train_data, train_data_labels)

def MLP_classifier_GridSearchCV(preprocessed_train_data, train_data_labels):
    """Train MLP classifier using given dataset and select best model with optimal parameters.

    Args:
        preprocessed_train_data (np array): preprocessed train data.
        train_data_labels (np array): train data labels.

    Returns:
        (dictionary): dictionary object contians information relate to different parameter combinations.

    """
    # setting search range for parameters
    parameters = { #'max_iter':[10, 50, 100, 150, 200] # another hyper parameter could be used for grid search (repetitive)
                  'hidden_layer_sizes':[(5), (5,5), (5,5,5), (5,5,5,5), (5,5,5,5,5), (10), 
                                        (10,10), (10,10,10), (10,10,10,10), (10,10,10,10,10)]} # 5,10; < 20neurons, <5 layers
    # tn f1 ... ... - get 4 accuracy scores (change format of grid search output)
    rfc = MLPClassifier()
    clf = GridSearchCV(rfc, parameters)
    # return the selected parameters for this estimator
    return clf.fit(preprocessed_train_data, train_data_labels).cv_results_


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