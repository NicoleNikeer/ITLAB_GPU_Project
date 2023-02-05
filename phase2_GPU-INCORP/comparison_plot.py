############################import useful libraries############################
import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pickle

from sklearn import datasets
from datetime import datetime
from matplotlib.pyplot import figure
from sklearn.svm import SVC 
from cuml.svm import SVC as SVC_gpu
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
from cuml.ensemble import RandomForestClassifier as RandomForestClassifier_gpu
from sklearn.naive_bayes import GaussianNB
from cuml.naive_bayes import GaussianNB as GaussianNB_gpu
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier


############################useful functions############################
def train_data(model, X, y):
    # *** function parameters *** #
    # model - the pre-defined learning model
    # X - the sample data for training
    # y - the label for classification prediction
    # *** ------------------- *** #
    clf = model
    clf.fit(X, y)

def run_svc(num_rounds, CPU_time_list, GPU_time_list, ratio_time_list, X, y):
    # *** function parameters *** #
    # * for current function
    # num_rounds - the number of the current round
    # CPU_time_list - the list to store the CPU runtime
    # GPU_time_list - the list to store the GPU runtime
    # ratio_time_list - the list to store the runtime ratio between CPU/GPU
    # for train_data function
    # X - the sample data for training
    # y - the label for classification prediction
    # *** ------------------- *** #
    # for svc
    clf_svc = SVC(kernel='poly', degree=2, gamma='auto', C=1)
    sklearn_time_svc = timeit.timeit(lambda: train_data(clf_svc, X, y), number=5)

    clf_svc = SVC_gpu(kernel='poly', degree=2, gamma='auto', C=1)
    cuml_time_svc = timeit.timeit(lambda: train_data(clf_svc, X, y), number=5)

    ratio_svc = sklearn_time_svc/cuml_time_svc

    print(f"""(svc) Average time of sklearn's {clf_svc.__class__.__name__}""", sklearn_time_svc, 's')
    print(f"""(svc) Average time of cuml's {clf_svc.__class__.__name__}""", cuml_time_svc, 's')
    print('(svc) Ratio between sklearn and cuml is', ratio_svc)

    CPU_time_list[num_rounds].append(sklearn_time_svc)
    GPU_time_list[num_rounds].append(cuml_time_svc)
    ratio_time_list[num_rounds].append(ratio_svc)


def run_rf(num_rounds, CPU_time_list, GPU_time_list, ratio_time_list, X, y):
    # *** function parameters *** #
    # for current function
    # num_rounds - the number of the current round
    # CPU_time_list - the list to store the CPU runtime
    # GPU_time_list - the list to store the GPU runtime
    # ratio_time_list - the list to store the runtime ratio between CPU/GPU
    # for train_data function
    # X - the sample data for training
    # y - the label for classification prediction
    # *** ------------------- *** #
    # for random forest
    clf_rf = RandomForestClassifier(max_features=1.0, n_estimators=40)
    sklearn_time_rf = timeit.timeit(lambda: train_data(clf_rf, X, y), number=5)

    clf_rf = RandomForestClassifier_gpu(max_features=1.0, n_estimators=40)
    cuml_time_rf = timeit.timeit(lambda: train_data(clf_rf, X, y), number=5)

    ratio_rf = sklearn_time_rf/cuml_time_rf

    print(f"""(rf) Average time of sklearn's {clf_rf.__class__.__name__}""", sklearn_time_rf, 's')
    print(f"""(rf) Average time of cuml's {clf_rf.__class__.__name__}""", cuml_time_rf, 's')
    print('(rf) Ratio between sklearn and cuml is', ratio_rf)

    CPU_time_list[num_rounds].append(sklearn_time_rf)
    GPU_time_list[num_rounds].append(cuml_time_rf)
    ratio_time_list[num_rounds].append(ratio_rf)

def run_nb(num_rounds, CPU_time_list, GPU_time_list, ratio_time_list, X, y):
    # *** function parameters *** #
    # for current function
    # num_rounds - the number of the current round
    # CPU_time_list - the list to store the CPU runtime
    # GPU_time_list - the list to store the GPU runtime
    # ratio_time_list - the list to store the runtime ratio between CPU/GPU
    # for train_data function
    # X - the sample data for training
    # y - the label for classification prediction
    # *** ------------------- *** #
    # for nb
    clf_nb = GaussianNB()
    sklearn_time_nb = timeit.timeit(lambda: train_data(clf_nb, X, y), number=5)

    start = datetime.now()
    for i in range(5):
        clf_nb = GaussianNB_gpu()# need to redeclare the GaussianNB_gpu() for multiple times???
        train_data(clf_nb, X, y)
    end = datetime.now()
    time_taken = (end - start).total_seconds()
    print(time_taken)
    cuml_time_nb = time_taken/5

    ratio_nb = sklearn_time_nb/cuml_time_nb

    print(f"""(nb) Average time of sklearn's {clf_nb.__class__.__name__}""", sklearn_time_nb, 's')
    print(f"""(nb) Average time of cuml's {clf_nb.__class__.__name__}""", cuml_time_nb, 's')
    print('(nb) Ratio between sklearn and cuml is', ratio_nb)

    CPU_time_list[num_rounds].append(sklearn_time_nb)
    GPU_time_list[num_rounds].append(cuml_time_nb)
    ratio_time_list[num_rounds].append(ratio_nb)

def run_boosting(num_rounds, CPU_time_list, GPU_time_list, ratio_time_list, X, y):
    # *** function parameters *** #
    # for current function
    # num_rounds - the number of the current round
    # CPU_time_list - the list to store the CPU runtime
    # GPU_time_list - the list to store the GPU runtime
    # ratio_time_list - the list to store the runtime ratio between CPU/GPU
    # for train_data function
    # X - the sample data for training
    # y - the label for classification prediction
    # *** ------------------- *** #
    # for boosting
    clf_ba = AdaBoostClassifier()
    sklearn_time_ba = timeit.timeit(lambda: train_data(clf_ba, X, y), number=5)

    clf_ba = LGBMClassifier()
    cuml_time_ba = timeit.timeit(lambda: train_data(clf_ba, X, y), number=5)

    ratio_ba = sklearn_time_ba/cuml_time_ba

    print(f"""(ba) Average time of sklearn's {clf_ba.__class__.__name__}""", sklearn_time_ba, 's')
    print(f"""(ba) Average time of cuml's {clf_ba.__class__.__name__}""", cuml_time_ba, 's')
    print('(ba) Ratio between sklearn and cuml is', ratio_ba)

    CPU_time_list[num_rounds].append(sklearn_time_ba)
    GPU_time_list[num_rounds].append(cuml_time_ba)
    ratio_time_list[num_rounds].append(ratio_ba)

def plot_fiture(current_graph_name, sample_or_feature, directory_to_store, CPU_time_list, GPU_time_list, ratio_time_list):
    # *** function parameters *** #
    # current_graph_name - indicate which learning model is used
    # sample_or_feature - mainly changing sample or feature
    # directory_to_store - directory name to store the graph
    # CPU_time_list - the list to store the CPU runtime
    # GPU_time_list - the list to store the GPU runtime
    # ratio_time_list - the list to store the runtime ratio between CPU/GPU
    # *** ------------------- *** #
    figure(figsize=(10, 10), dpi=80)

    plt.plot(x_axis, CPU_time_list[0], label="CPU 50 features", color="#F39C12", linestyle="--")
    plt.plot(x_axis, GPU_time_list[0], label="GPU 50 features", color="#F39C12")
    plt.plot(x_axis, CPU_time_list[1], label="CPU 100 features", color="#27AE60", linestyle="--")
    plt.plot(x_axis, GPU_time_list[1], label="GPU 100 features", color="#27AE60")
    plt.plot(x_axis, CPU_time_list[2], label="CPU 150 features", color="#2980B9", linestyle="--")
    plt.plot(x_axis, GPU_time_list[2], label="GPU 150 features", color="#2980B9")
    plt.plot(x_axis, CPU_time_list[3], label="CPU 200 features", color="#8E44AD", linestyle="--")
    plt.plot(x_axis, GPU_time_list[3], label="GPU 200 features", color="#8E44AD")
    plt.plot(x_axis, CPU_time_list[4], label="CPU 250 features", color="#C0392B", linestyle="--")
    plt.plot(x_axis, GPU_time_list[4], label="GPU 250 features", color="#C0392B")
    plt.legend(loc='upper right')
    if current_graph_name == "ba":
        plt.title("MSAP (CPU) classifier: Adaboost vs. (GPU) classifier: LightGBM")
    else:
        plt.title("MSAP (CPU) classifier: " + current_graph_name + " from sklearn vs. (GPU) classifier: " + current_graph_name  + " from cuml")
    plt.xlabel(sample_or_feature + ' Size')
    plt.ylabel('Computation Time(s)')
    plt.savefig(directory_to_store + current_graph_name + '_compare_' + sample_or_feature +'_plot.png')
    plt.cla()

    figure(figsize=(10, 10), dpi=80)

    plt.plot(x_axis, ratio_time_list[0], label="50 features", color="#F39C12")
    plt.plot(x_axis, ratio_time_list[1], label="100 features", color="#27AE60")
    plt.plot(x_axis, ratio_time_list[2], label="150 features", color="#2980B9")
    plt.plot(x_axis, ratio_time_list[3], label="200 features", color="#8E44AD")
    plt.plot(x_axis, ratio_time_list[4], label="250 features", color="#C0392B")
    plt.legend(loc='upper right')
    if current_graph_name == "ba":
        plt.title("MSAP (CPU) classifier: Adaboost vs. (GPU) classifier: LightGBM")
    else:
        plt.title("MSAP (CPU) classifier: " + current_graph_name + " from sklearn vs. (GPU) classifier: " + current_graph_name  + " from cuml")
    plt.xlabel(sample_or_feature + ' Size')
    plt.ylabel('Computation Time(s)')
    plt.savefig(directory_to_store + current_graph_name + '_ratio_' + sample_or_feature + '_plot.png')
    plt.cla()


############################main function############################
if __name__ == '__main__':

    # store all the results in a dictionary
    final_result = {}


    #----------------------------change sample size----------------------------#
    # declare lists to store CPU and GPU runtimes
    svc_average_time_sklearn_all = [ [], [], [], [], [] ]
    svc_average_time_cuml_all = [ [], [], [], [], [] ]
    svc_ratio_between_all = [ [], [], [], [], [] ]
    rf_average_time_sklearn_all = [ [], [], [], [], [] ]
    rf_average_time_cuml_all = [ [], [], [], [], [] ]
    rf_ratio_between_all = [ [], [], [], [], [] ]
    nb_average_time_sklearn_all = [ [], [], [], [], [] ]
    nb_average_time_cuml_all = [ [], [], [], [], [] ]
    nb_ratio_between_all = [ [], [], [], [], [] ]
    ba_average_time_cpu_all = [ [], [], [], [], [] ]
    ba_average_time_gpu_all = [ [], [], [], [], [] ]
    ba_ratio_between_all = [ [], [], [], [], [] ]
    # initial number of features
    num_features = 50

    # five rounds, increase feature numbers (50, 100, 150, 200, 250, 300)
    for f_index in range(5):
        sample_numbers = 0 # reinitialize sample size
        num_features = 50 * (f_index + 1) # incrase feature number
        
        while sample_numbers < 5001:
            sample_numbers += 500 # update sample size
            print("(change sample size: ", sample_numbers, num_features,")") # print  indication for rounds
            X, y  = datasets.make_classification(n_samples=sample_numbers, n_features = num_features) # generate sample data

            # some cuML models require the input to be np.float32
            X = X.astype(np.float32)
            y = y.astype(np.float32)

            # *** train different learning model and record runtime *** #
            # for svc
            run_svc(f_index, svc_average_time_sklearn_all, svc_average_time_cuml_all, svc_ratio_between_all, X, y)
            # for random forest
            run_rf(f_index, rf_average_time_sklearn_all, rf_average_time_cuml_all, rf_ratio_between_all, X, y)
            # for nb
            run_nb(f_index, nb_average_time_sklearn_all, nb_average_time_cuml_all, nb_ratio_between_all, X, y)
            # for boosting
            run_boosting(f_index, ba_average_time_cpu_all, ba_average_time_gpu_all, ba_ratio_between_all, X, y)

    # define x-axis for plotting
    x_axis = [ i for i in range(500, 5001, 500)]

    # *** generate plots for changing sample size *** #
    # plot svc
    plot_fiture("svc", "Sample", "./update_sample/", svc_average_time_sklearn_all, svc_average_time_cuml_all, svc_ratio_between_all)
    # plot rf
    plot_fiture("rf", "Sample", "./update_sample/", rf_average_time_sklearn_all, rf_average_time_cuml_all, rf_ratio_between_all)
    # plot nb
    plot_fiture("nb", "Sample", "./update_sample/", nb_average_time_sklearn_all, nb_average_time_cuml_all, nb_ratio_between_all)
    # plot ba
    plot_fiture("ba", "Sample", "./update_sample/", ba_average_time_cpu_all, ba_average_time_gpu_all, ba_ratio_between_all)

    ### Saving results ###
    final_result["change_sample_size"] = {}
    final_result["change_sample_size"]["svc"] = [copy.deepcopy(svc_average_time_sklearn_all), copy.deepcopy(svc_average_time_cuml_all), copy.deepcopy(svc_ratio_between_all)]
    final_result["change_sample_size"]["rf"] = [copy.deepcopy(rf_average_time_sklearn_all), copy.deepcopy(rf_average_time_cuml_all), copy.deepcopy(rf_ratio_between_all)]
    final_result["change_sample_size"]["nb"] = [copy.deepcopy(nb_average_time_sklearn_all), copy.deepcopy(nb_average_time_cuml_all), copy.deepcopy(nb_ratio_between_all)]
    final_result["change_sample_size"]["ba"] = [copy.deepcopy(ba_average_time_cpu_all), copy.deepcopy(ba_average_time_gpu_all), copy.deepcopy(ba_ratio_between_all)]


    #----------------------------change feature size----------------------------#
    # declare lists to store CPU and GPU runtimes
    svc_average_time_sklearn_all = [ [], [], [], [], [] ]
    svc_average_time_cuml_all = [ [], [], [], [], [] ]
    svc_ratio_between_all = [ [], [], [], [], [] ]
    rf_average_time_sklearn_all = [ [], [], [], [], [] ]
    rf_average_time_cuml_all = [ [], [], [], [], [] ]
    rf_ratio_between_all = [ [], [], [], [], [] ]
    nb_average_time_sklearn_all = [ [], [], [], [], [] ]
    nb_average_time_cuml_all = [ [], [], [], [], [] ]
    nb_ratio_between_all = [ [], [], [], [], [] ]
    ba_average_time_sklearn_all = [ [], [], [], [], [] ]
    ba_average_time_cuml_all = [ [], [], [], [], [] ]
    ba_ratio_between_all = [ [], [], [], [], [] ]
    # initial number of samples
    num_samples = 50

    # five rounds, increase sample numbers (50, 100, 150, 200, 250, 300)
    for f_index in range(5):
        feature_numbers = 0 # reinitialize feature size
        num_samples = 50 * (f_index + 1) # incrase sample number

        while feature_numbers < 5001:
            feature_numbers += 500 # update feature size
            print("(change feature size: ", num_samples, feature_numbers,")")# print  indication for rounds
            X, y  = datasets.make_classification(n_samples=num_samples, n_features=feature_numbers) # generate sample data

            # some cuML models require the input to be np.float32
            X = X.astype(np.float32)
            y = y.astype(np.float32)

            # *** train different learning model and record runtime *** #
            # for svc
            run_svc(f_index, svc_average_time_sklearn_all, svc_average_time_cuml_all, svc_ratio_between_all, X, y)
            # for random forest
            run_rf(f_index, rf_average_time_sklearn_all, rf_average_time_cuml_all, rf_ratio_between_all, X, y)
            # for nb
            run_nb(f_index, nb_average_time_sklearn_all, nb_average_time_cuml_all, nb_ratio_between_all, X, y)
            # for boosting
            run_boosting(f_index, ba_average_time_cpu_all, ba_average_time_gpu_all, ba_ratio_between_all, X, y)
    
    # define x-axis for plotting
    # this is already defined before

    # *** generate plots for changing sample size *** #
    # plot svc
    plot_fiture("svc", "Feature", "./update_feature/", svc_average_time_sklearn_all, svc_average_time_cuml_all, svc_ratio_between_all)
    # plot rf
    plot_fiture("rf", "Feature", "./update_feature/", rf_average_time_sklearn_all, rf_average_time_cuml_all, rf_ratio_between_all)
    # plot nb
    plot_fiture("nb", "Feature", "./update_feature/", nb_average_time_sklearn_all, nb_average_time_cuml_all, nb_ratio_between_all)
    # plot ba
    plot_fiture("ba", "Feature", "./update_feature/", ba_average_time_cpu_all, ba_average_time_gpu_all, ba_ratio_between_all)

    ### Saving results ###
    final_result["change_feature_size"] = {}
    final_result["change_feature_size"]["svc"] = [copy.deepcopy(svc_average_time_sklearn_all), copy.deepcopy(svc_average_time_cuml_all), copy.deepcopy(svc_ratio_between_all)]
    final_result["change_feature_size"]["rf"] = [copy.deepcopy(rf_average_time_sklearn_all), copy.deepcopy(rf_average_time_cuml_all), copy.deepcopy(rf_ratio_between_all)]
    final_result["change_feature_size"]["nb"] = [copy.deepcopy(nb_average_time_sklearn_all), copy.deepcopy(nb_average_time_cuml_all), copy.deepcopy(nb_ratio_between_all)]
    final_result["change_feature_size"]["ba"] = [copy.deepcopy(ba_average_time_sklearn_all), copy.deepcopy(ba_average_time_cuml_all), copy.deepcopy(ba_ratio_between_all)]


    # dump output to a pickle file
    with open('./outputs/comparison_plot_results.pickle', 'wb') as handle:
        pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #---------pickle file access instructions---------#
    # import pickle

    # your_data = {'foo': 'bar'}

    # # Store data (serialize)
    # with open('filename.pickle', 'wb') as handle:
    #     pickle.dump(your_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Load data (deserialize)
    # with open('filename.pickle', 'rb') as handle:
    #     unserialized_data = pickle.load(handle)

    # print(your_data == unserialized_data)
    #---------pickle file access instructions---------#