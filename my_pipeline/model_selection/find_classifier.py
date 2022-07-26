"""Finding the best classifier.

Authors:
    Keer Ni - knni@ucdavis.edu

"""

"""Perform classification.

1. SVC
2. RandomForest
3. MLP

"""

# get the target labels

from sklearn import svm
from sklearn.model_selection import GridSearchCV

def SVC_classifier(im_train_data):
    """Detect outlier using IsolationForest.

    Args:
        im_train_data (list of lists): imputed train data.

    Returns:
        (list): indices of detected outliers.

    """
    clf = IsolationForest(random_state=0).fit(im_train_data)

    is_outlier = list(clf.predict(im_train_data))

    return [i for i, x in enumerate(is_outlier) if x != 1]