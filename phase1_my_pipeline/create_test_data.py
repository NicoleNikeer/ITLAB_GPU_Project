"""Separate the given test data to train-test split (80%-20%).

Authors:
    Keer Ni - knni@ucdavis.edu

"""

"""
In Command Line:

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

"""

import random
import copy
import numpy as np
import pandas as pd

def my_train_test_split(dataset):
    """Split a given test data to 80% train and 20% test.

    Args:
        data (list of lists): all data for the experiment.

    Returns:
        train_data (list), test_data (list): 80%, 20% of the input/original data set.

    """
    total_rows = len(dataset.data)
    num_separation_row = int(1*total_rows)
    # randomly select 80% of data to be test data
    num_test_data = set(random.sample(range(len(dataset.data)), num_separation_row))

    # grab train_data and labels/targets
    train_data = [list(x) for i, x in enumerate(dataset.data) if i in num_test_data]
    train_data_labels = [x for i, x in enumerate((dataset.target).tolist()) if i in num_test_data]
    # grab test_data and labels/targets
    test_data = [list(x) for i, x in enumerate(dataset.data) if i not in num_test_data]
    test_data_labels = [x for i, x in enumerate((dataset.target).tolist()) if i not in num_test_data]

    return train_data, train_data_labels, test_data, test_data_labels


def my_random_remove(train_data):
    """Randomly remove data in train data.

    Args:
        train_data (list of lists): train data for the the experiment.

    Returns:
        or_trian_data (list of lists): train data with no random removal.
        re_train_data (list of lists): train data with random removal.

    """

    # random state/seed? -> make the result the same

    nan = np.nan

    ran_num_rmv = random.randint(0, 1) # use numpy random generator (specify random seed)
    re_train_data = []
    or_train_data = copy.deepcopy(train_data)
    for a_row in train_data:
        # remove random data
        if ran_num_rmv == 1:
            ran_index_rmv = random.randint(0, len(train_data[0])-1)
            a_row[ran_index_rmv] = nan
            re_train_data.append(a_row)
        else:
            re_train_data.append(a_row)

    return or_train_data, re_train_data


if __name__ == '__main__':
    """
        Using the iris dataset from scikit learn.
        Saving an array to a binary file in NumPy .npy format.

        Only need to run this .py for once to create a dataset with randomly removed data points in current work space.
    """
    from sklearn import datasets
    iris = datasets.load_iris()

    train_data, train_data_labels, test_data, test_data_labels = my_train_test_split(iris)
    original_train_data, removed_train_data = my_random_remove(train_data)

    npremoved_train_data = np.array(removed_train_data)
    nporiginal_train_data = np.array(original_train_data)
    nptrain_data_labels = np.array(train_data_labels)
    nptest_data = np.array(test_data)
    nptest_data_labels = np.array(test_data_labels)

    print(removed_train_data)
    print(original_train_data)

    print(npremoved_train_data)
    print(nporiginal_train_data)

    print(nptrain_data_labels)
    print(nptest_data_labels)

    with open('create_test_data.npy', 'wb') as f:
       np.save(f, npremoved_train_data)
       np.save(f, nporiginal_train_data)
       np.save(f, nptrain_data_labels)
       np.save(f, test_data)
       np.save(f, nptest_data_labels)

    # with open('npremoved_train_data.npy', 'rb') as f:
    #     a = np.load(f)
    # print(a)

# store the train data & train label in a csv file
    i = 0
    a = np.zeros(shape=(150,5))
    for a_row in npremoved_train_data:
        a[i] = np.append(npremoved_train_data[i], nptrain_data_labels[i])
        i += 1
    DF = pd.DataFrame(a)
    DF.to_csv("iris_data_ori.csv")
