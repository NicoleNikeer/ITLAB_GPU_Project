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


if __name__ == '__main__':

    # 1. Read the test data from the .csv file
    # NOTICE: here, the data (in .csv file) already have data randomly removed
    df = pd.read_csv('iris_data.csv')

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