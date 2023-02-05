import numpy as np
import timeit
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd

from sklearn.svm import SVC 
from cuml.svm import SVC as SVC_gpu
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
from cuml.ensemble import RandomForestClassifier as RandomForestClassifier_gpu
from sklearn.naive_bayes import GaussianNB
from cuml.naive_bayes import GaussianNB as GaussianNB_gpu
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier



if __name__ == '__main__':

    # generate the dataset
    X, y  = datasets.make_classification(n_samples=5000, n_features=500)

    # some cuML models require the input to be np.float32 [TOBE explore more]
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # split for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    # for svc
    print("----------------sklearn svc results----------------")
    clf_sklearn_svc = SVC(kernel='poly', degree=2, gamma='auto', C=1)
    clf_sklearn_svc.fit(X_train, y_train)
    y_pred = clf_sklearn_svc.predict(X_test)
    # check accuracy
    accuracy_sklearn_svc = accuracy_score(y_pred, y_test)
    print('Sklearn SVC Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    # check confusion matrix
    cm_sklearn_svc = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n\n', cm_sklearn_svc)
    print('\nTrue Positives(TP) = ', cm_sklearn_svc[0,0])
    print('\nTrue Negatives(TN) = ', cm_sklearn_svc[1,1])
    print('\nFalse Positives(FP) = ', cm_sklearn_svc[0,1])
    print('\nFalse Negatives(FN) = ', cm_sklearn_svc[1,0])
    # plot confusion matrix
    df_cm = pd.DataFrame(cm_sklearn_svc, range(2), range(2))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},cmap='Blues', fmt='g') # font size
    plt.savefig('cm_sklearn_svc.png')
    # classification metrices
    cr_sklearn_svc = classification_report(y_test, y_pred)
    print(classification_report(y_test, y_pred))



    # clf_svc = SVC_gpu(kernel='poly', degree=2, gamma='auto', C=1)
    # cuml_time_svc = timeit.timeit(lambda: train_data(clf_svc, X, y), number=5)

    # ratio_svc = sklearn_time_svc/cuml_time_svc

    # print(f"""(svc) Average time of sklearn's {clf_svc.__class__.__name__}""", sklearn_time_svc, 's')
    # print(f"""(svc) Average time of cuml's {clf_svc.__class__.__name__}""", cuml_time_svc, 's')
    # print('(svc) Ratio between sklearn and cuml is', ratio_svc)

    # svc_average_time_sklearn_all[f_index].append(sklearn_time_svc)
    # svc_average_time_cuml_all[f_index].append(cuml_time_svc)
    # svc_ratio_between_all[f_index].append(ratio_svc)




    # # for random forest
    # clf_rf = RandomForestClassifier(max_features=1.0, n_estimators=40)
    # sklearn_time_rf = timeit.timeit(lambda: train_data(clf_rf, X, y), number=5)

    # clf_rf = RandomForestClassifier_gpu(max_features=1.0, n_estimators=40)
    # cuml_time_rf = timeit.timeit(lambda: train_data(clf_rf, X, y), number=5)

    # ratio_rf = sklearn_time_rf/cuml_time_rf

    # print(f"""(rf) Average time of sklearn's {clf_rf.__class__.__name__}""", sklearn_time_rf, 's')
    # print(f"""(rf) Average time of cuml's {clf_rf.__class__.__name__}""", cuml_time_rf, 's')
    # print('(rf) Ratio between sklearn and cuml is', ratio_rf)

    # rf_average_time_sklearn_all[f_index].append(sklearn_time_rf)
    # rf_average_time_cuml_all[f_index].append(cuml_time_rf)
    # rf_ratio_between_all[f_index].append(ratio_rf)

    # # for nb
    # clf_nb = GaussianNB()
    # sklearn_time_nb = timeit.timeit(lambda: train_data(clf_nb, X, y), number=5)

    # start = datetime.now()
    # for i in range(5):
    #     clf_nb = GaussianNB_gpu()# need to redeclare the GaussianNB_gpu() for multiple times???
    #     train_data(clf_nb, X, y)
    # end = datetime.now()
    # time_taken = (end - start).total_seconds()
    # print(time_taken)
    # cuml_time_nb = time_taken/5

    # ratio_nb = sklearn_time_nb/cuml_time_nb

    # print(f"""(nb) Average time of sklearn's {clf_nb.__class__.__name__}""", sklearn_time_nb, 's')
    # print(f"""(nb) Average time of cuml's {clf_nb.__class__.__name__}""", cuml_time_nb, 's')
    # print('(nb) Ratio between sklearn and cuml is', ratio_nb)

    # nb_average_time_sklearn_all[f_index].append(sklearn_time_nb)
    # nb_average_time_cuml_all[f_index].append(cuml_time_nb)
    # nb_ratio_between_all[f_index].append(ratio_nb)

    # # for boosting
    # clf_ba = AdaBoostClassifier()
    # sklearn_time_ba = timeit.timeit(lambda: train_data(clf_ba, X, y), number=5)

    # clf_ba = LGBMClassifier()
    # cuml_time_ba = timeit.timeit(lambda: train_data(clf_ba, X, y), number=5)

    # ratio_ba = sklearn_time_ba/cuml_time_ba

    # print(f"""(ba) Average time of sklearn's {clf_ba.__class__.__name__}""", sklearn_time_ba, 's')
    # print(f"""(ba) Average time of cuml's {clf_ba.__class__.__name__}""", cuml_time_ba, 's')
    # print('(ba) Ratio between sklearn and cuml is', ratio_ba)

    # ba_average_time_sklearn_all[f_index].append(sklearn_time_ba)
    # ba_average_time_cuml_all[f_index].append(cuml_time_ba)
    # ba_ratio_between_all[f_index].append(ratio_ba)

