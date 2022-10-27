#from data_preprocessing import MissForest_impute
from random import random
import pandas as pd
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import copy
import sys
import warnings
import numpy as np
import pickle
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base # noqa
from missingpy import MissForest

from create_test_data import my_train_test_split
from data_preprocessing import MissForest_impute, KNN_impute
from data_preprocessing import MinMax_scale, Standardize_scale
from find_classifier import (SVC_classifier_GridSearchCV, 
                             RandomForest_classifier_GridSearchCV, 
                             MLP_classifier_GridSearchCV)

# for pr curve
import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import make_blobs
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import plot_precision_recall_curve


# for roc curve
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_roc_curve



if __name__ == '__main__':

    # 1. Read the test data from the .csv file
    # NOTICE: here, the data (in .csv file) already have data randomly removed
    df = pd.read_csv('iris_data_fin.csv')
    df_ori = pd.read_csv('iris_data_ori.csv')

    # 2. Convert the data to numpy array
    # In the np array, nan is the data type for the missing/randomly removed data
    data = df.to_numpy()
    last_column_index = len(data[0]) - 1
    # Grab the left-most column for y (testing data)
    y = [a_row[last_column_index] for a_row in data]
    y_np = np.array(y)
    # Grab the other column for X (training data)
    x = [a_row[0:last_column_index] for a_row in data]
    x_np = np.array(x)
    # Split the data for train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(x_np, y_np, random_state=42)

    # plot hisgram for the variables
    df.hist(figsize=(16,16), grid=False)
    plt.savefig('histogram_fin.png')
    plt.clf()
    df_ori.hist(figsize=(16,16), grid=False)
    plt.savefig('histogram_ori.png')

    # 3. Impute the test data
    # //?// how to decide should choose MissForest or KNN
    X_train_im = MissForest_impute(X_train)
    # Find the average of the imputed data for each column. 
    # (using np.mean, axis=0 is for columns, axis=1 is for rows)
    X_train_averages = np.mean(X_train_im, axis=0)
    # Use the averages for missing values in test data.
    X_test_im = np.copy(X_test)
    for row in X_test_im:
        for j in range(len(X_test_im[0])):
            if np.isnan(row[j]):
                 row[j] = X_train_averages[j]

    # 4. Plot and output the TSNE graph for visualization
    # Collect all the data.
    X_all = np.append(X_train_im, X_test_im, axis=0)
    Y_all = np.append(Y_train, Y_test)
    # Do the TSNE plot.
    tsne = TSNE(n_components=2, verbose=1, random_state=42, learning_rate='auto', init='random')
    z = tsne.fit_transform(X_all)
    df = pd.DataFrame()
    df["y"] = Y_all
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=df).set(title="Iris data T-SNE projection")
    plt.savefig('iris_TSNE_9141(7).png') # could use current time
    plt.close()

    # 4). Plot and output the TSNE graph for visualization
    # After scaling the data.
    # //?// should we obtain the TSNE graph after scaling or not
    X_all_sc = MinMax_scale(X_all)
    tsne = TSNE(n_components=2, verbose=1, random_state=42, learning_rate='auto', init='random')
    z = tsne.fit_transform(X_all_sc)
    df = pd.DataFrame()
    df["y"] = Y_all
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=df).set(title="Iris data T-SNE projection")
    plt.savefig('iris_TSNEsc_9141(6).png')

    # 5. Four different ways of imputing&scaling the dataset
    # minmax scaling & missforest imputation
    X_all_mm = MinMax_scale(X_train)
    X_all_mf_mm = MissForest_impute(X_all_mm)
    X_train_averages = np.mean(X_all_mf_mm, axis=0)
    X_test_mf_mm = MinMax_scale(np.copy(X_test))
    for row in X_test_mf_mm:
        for j in range(len(X_test_mf_mm[0])):
            if np.isnan(row[j]):
                 row[j] = X_train_averages[j]
    # minmax scaling & knn 
    X_all_knn_mm = KNN_impute(X_all_mm)
    X_train_averages = np.mean(X_all_knn_mm, axis=0)
    X_test_knn_mm = MinMax_scale(np.copy(X_test))
    for row in X_test_knn_mm:
        for j in range(len(X_test_knn_mm[0])):
            if np.isnan(row[j]):
                 row[j] = X_train_averages[j]
    # standardize scaling & missforest imputation
    X_all_sd = Standardize_scale(X_train)
    X_all_mf_sd = MissForest_impute(X_all_sd)
    X_train_averages = np.mean(X_all_mf_sd, axis=0)
    X_test_mf_sd = Standardize_scale(np.copy(X_test))
    for row in X_test_mf_sd:
        for j in range(len(X_test_mf_sd[0])):
            if np.isnan(row[j]):
                 row[j] = X_train_averages[j]
    # standardize scaling & knn imputation
    X_all_knn_sd = KNN_impute(X_all_sd)
    X_train_averages = np.mean(X_all_knn_sd, axis=0)
    X_test_knn_sd = Standardize_scale(np.copy(X_test))
    for row in X_test_knn_sd:
        for j in range(len(X_test_knn_sd[0])):
            if np.isnan(row[j]):
                 row[j] = X_train_averages[j]
    # # TODO: Compare and see which imputing method is better/closer to the original data set.
    # X_train_MFim = MissForest_impute(X_train)
    # X_train_KNNim = KNN_impute(X_train)
    # # Analysis by finding differences of the numpy array
    # #  //?// how to find the difference - using norm/standard error?
    # iris = datasets.load_iris()
    # X_train_original = np.array([list(x) for i, x in enumerate(iris['data']) if i <= 99])
    # print(X_train_original)
    # print(X_train_MFim)
    # print(X_train_KNNim)

    # 6. Do the Grid Search and store the best classifier
    X_all_list = [X_all_mf_mm, X_all_mf_sd, X_all_knn_mm, X_all_knn_sd] # store all processed data in a list
    X_pipeline_all_list = ["MissForest & MinMax", "MissForest & Standardize", "KNN & MinMax", "KNN & Standardize"] # store name of the pipelines list
    X_test_all_list = [X_test_mf_mm, X_test_mf_sd, X_test_knn_mm, X_test_knn_sd] # store all processed test data in a list
    gridsearch_list = [SVC_classifier_GridSearchCV, RandomForest_classifier_GridSearchCV, MLP_classifier_GridSearchCV] # store all gridsearch in a list
    best_classifiers = {'y_train': Y_train, 'y_test': Y_test} # use a dictionary to store best classifiers
    count = 0
    for i in range(len(X_all_list)):
        for k in range(len(gridsearch_list)):
            after_search = gridsearch_list[k](X_all_list[i], Y_train)
            max_mean = np.max(after_search['mean_test_score'])
            for j in range(len(after_search['mean_test_score'])):
                if after_search['mean_test_score'][j] == max_mean:
                    best_classifiers[count] = {}
                    best_classifiers[count]['parameters'] = after_search['params'][j]
                    best_classifiers[count]['classifier'] = gridsearch_list[k]
                    best_classifiers[count]['test_score'] = after_search['mean_test_score'][j]
                    best_classifiers[count]['x_train'] = X_all_list[i]
                    best_classifiers[count]['x_test'] = X_test_all_list[i]
                    best_classifiers[count]['data preprocess pipeline methods'] = X_pipeline_all_list[i]
                    count += 1
        print('finish one search')

    # 7. Find and plot the confusion matrix
    best_result = best_classifiers[0]
    best_X_train = best_result['x_train']
    best_Y_train = best_classifiers['y_train']
    best_X_test = best_result['x_test']
    best_Y_test = best_classifiers['y_test']
    best_parameters = best_result['parameters']
    classifier = svm.SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], 
                         gamma=best_parameters['gamma']).fit(best_X_train, best_Y_train)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier,
            best_X_test,
            best_Y_test,
            display_labels=["c1", "c2"],
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.savefig('cm(1).png')

    # # TODO: Tidy up and comments the code for PR and ROC curves
    # outputting the pr curve
    models = [SVC(kernel='linear', C=1.0, probability=True, random_state=12345).fit(X_test_mf_mm, Y_test)]
    fig, ax = plt.subplots()
    for m in models:
        plot_precision_recall_curve(m, X_test_mf_mm, Y_test, ax=ax)
    ax.axhline(0.9, c='w', ls="--", lw=1, alpha=0.5)
    ax.axvline(0.9, c='w', ls="--", lw=1, alpha=0.5)
    ax.set_title("PR Curve");
    fig.savefig('result_prc_test.png')

    plt.clf()

    FOLDS = 5

    X, y = X_all_mf_sd, Y_train
    #X, y = X_test, Y_test
    f, axes = plt.subplots(1, 2, figsize=(10,5))

    axes[0].scatter(X[y==0,0], X[y==0,1], color='blue', s=2, label='y=0')
    axes[0].scatter(X[y!=0,0], X[y!=0,1], color='red', s=2, label='y=1')
    axes[0].set_xlabel('Sepal length')
    axes[0].set_ylabel('Sepal width')
    axes[0].legend(loc='lower left', fontsize='small')

    k_fold = KFold(n_splits=FOLDS, shuffle=True, random_state=12345)
    predictor = SVC(kernel='linear', C=1.0, probability=True, random_state=12345)

    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X)):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, pred_proba[:,1])
        lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
        axes[1].step(recall, precision, label=lab)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])

    y_real = numpy.concatenate(y_real)
    y_proba = numpy.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'Overall AUC=%.4f' % (auc(recall, precision))
    axes[1].step(recall, precision, label=lab, lw=2, color='black')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='lower left', fontsize='small')

    f.tight_layout()
    f.savefig('result_prc.png')

    plt.clf()

    # outputting the ROC curve
    models = [SVC(kernel='linear', C=1.0, probability=True, random_state=12345).fit(X_test_mf_mm, Y_test)]
    fig, ax = plt.subplots()
    for m in models:
        plot_roc_curve(m, X_test_mf_mm, Y_test, ax=ax)
    ax.axhline(0.9, c='w', ls="--", lw=1, alpha=0.5)
    ax.axvline(0.9, c='w', ls="--", lw=1, alpha=0.5)
    ax.set_title("ROC Curve");
    fig.savefig('result_roc_test.png')
    plt.clf()

    # Run classifier with cross-validation and plot ROC curves
    iris = datasets.load_iris()
    X = X_all_mf_sd
    y = Y_train
    #X = X_test
    #y = Y_test
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    # # Add noisy features
    # random_state = np.random.RandomState(0)
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    cv = StratifiedKFold(n_splits=5)
    classifier = svm.SVC(kernel='linear', C=1.0, probability=True, random_state=12345)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    plt.savefig('result_roc_test_fin.png')

    # 8. Store the outputs in a pickle file
    with open('my_pipeline_output.pkl', 'wb') as file:
        pickle.dump(best_classifiers, file)




    # ------------------------------------------NOTES------------------------------------------

    # Todo(1):
    # - Store every not reproducable things in pickle file
    # e.g. hyper parameter, confusion matrix, // use a dictionary to store these for every model
    #      curves (ROC curve, PR curve) for the best model

    # TSNE (use) - best model imputation method (for audiences to see)
    #            - try every imputation method (for yourself to see) 
    #               // hope no difference; otherwise there might be some assumption in the data 
    #               imputation/scaling that is wrong that leads to the difference 
    #               // make understanding of the data set better

    # Different ML model has different assumptions:
    # - SVM: (1) assume/best fit for normally distributed data; 
    #        (2) differentiate data base on distance (std = 1); 
    #        (3) best for data with lower dimension/fewer features (curse of dimensionality).
    # - RandomForest: (1) best fit for ranked data; 
    #                 (2) no require for distance; 
    #                 (3) best for data which has distinct features (e.g. useful in bio since 
    #                     distinct features determine by expertises).
    # - MLP: (1) no assumption for data; 
    #        (2) best for large data set (data set with many sample/trials); 
    #        (3) not good for small data set.

    # Todo(2):
    # - grid search: change scoring function
    # - parsing to a more readable dictionary
    # - confusion matrix plot

    # MSAP: using stratified K-folds, so each split has approximately same number of 
    #       true-positives and true-negatives.