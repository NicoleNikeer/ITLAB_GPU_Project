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

def train_test_split(data):
    """Split a given test data to 80% train and 20% test.

    Args:
        data (list of lists): all data for the experiment.

    Returns:
        train_data (list), test_data (list): 80%, 20% of the input/original data set.

    """
    total_rows = len(data)
    num_separation_row = int(0.8*total_rows)
    # randomly select 80% of data to be test data
    num_test_data = set(random.sample(range(len(data)), num_separation_row))

    # grab train_data
    train_data = [list(x) for i, x in enumerate(data) if i in num_test_data]
    # grab test_data
    test_data = [list(x) for i, x in enumerate(data) if i not in num_test_data]

    return train_data, test_data


def random_remove(train_data):
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
    from sklearn import datasets
    iris = datasets.load_iris()

    train_data, test_data = train_test_split(iris.data)
    or_train_data, re_train_data = random_remove(train_data)

    npre_train_data = np.array(re_train_data)
    npor_train_data = np.array(or_train_data)

    print(re_train_data)
    print(or_train_data)

    print(npre_train_data)
    print(npor_train_data)

#    with open('npre_train_data.npy', 'wb') as f:
#        np.save(f, npre_train_data)
    #    np.save(f, npor_train_data)

    with open('npre_train_data.npy', 'rb') as f:
        a = np.load(f)
    print(a)

# store as a csv file?