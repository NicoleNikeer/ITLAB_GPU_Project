############################import useful libraries############################
import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pickle
import itertools

from statistics import mean
from statistics import pstdev
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
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

def run_svc(num_rounds, size_number, CPU_time_list, GPU_time_list, ratio_time_list, X, y):
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
    # do Grid Search for svc
    parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 
                  'C':[1, 10, 100], 'degree':[1,2,4,8], 'gamma':('scale', 'auto')}
    # parameters = {'kernel':('linear', 'poly'), 
    #               'C':[1, 10], 'degree':[1,2], 'gamma':('scale', 'auto')}
    # all parameters combinations:
    all_combination_parameters = itertools.product(*parameters.values())

    for a_combination_parameter in all_combination_parameters:
        # for svc
        cur_kernel = a_combination_parameter[0]
        cur_C = a_combination_parameter[1]
        cur_degree = a_combination_parameter[2]
        cur_gamma = a_combination_parameter[3]
        clf_svc = SVC(kernel=cur_kernel, degree=cur_degree, gamma=cur_gamma, C=cur_C)
        sklearn_time_svc = timeit.timeit(lambda: train_data(clf_svc, X, y), number=5)

        clf_svc = SVC_gpu(kernel=cur_kernel, degree=cur_degree, gamma=cur_gamma, C=cur_C)
        cuml_time_svc = timeit.timeit(lambda: train_data(clf_svc, X, y), number=5)

        ratio_svc = sklearn_time_svc/cuml_time_svc

        print("Current parameters - kernel: " + cur_kernel + " C: " + str(cur_C) + " degree: " + str(cur_degree) + " gamma: " + cur_gamma)
        print(f"""(svc) Average time of sklearn's {clf_svc.__class__.__name__}""", sklearn_time_svc, 's')
        print(f"""(svc) Average time of cuml's {clf_svc.__class__.__name__}""", cuml_time_svc, 's')
        print('(svc) Ratio between sklearn and cuml is', ratio_svc)

        if num_rounds not in CPU_time_list:
            # initialize a list in a given dictionary key
            CPU_time_list[num_rounds] = {}
            GPU_time_list[num_rounds] = {}
            ratio_time_list[num_rounds] = {}
            CPU_time_list[num_rounds][(size_number, (cur_kernel, cur_C, cur_degree, cur_gamma))] = (sklearn_time_svc)
            GPU_time_list[num_rounds][(size_number, (cur_kernel, cur_C, cur_degree, cur_gamma))] = (cuml_time_svc)
            ratio_time_list[num_rounds][(size_number, (cur_kernel, cur_C, cur_degree, cur_gamma))] = (ratio_svc)
        else:
            # append to an existing list
            CPU_time_list[num_rounds][(size_number, (cur_kernel, cur_C, cur_degree, cur_gamma))] = (sklearn_time_svc)
            GPU_time_list[num_rounds][(size_number, (cur_kernel, cur_C, cur_degree, cur_gamma))] = (cuml_time_svc)
            ratio_time_list[num_rounds][(size_number, (cur_kernel, cur_C, cur_degree, cur_gamma))] = (ratio_svc)

def run_rf(num_rounds, size_number, CPU_time_list, GPU_time_list, ratio_time_list, X, y):
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
    # do Grid Search for rf
    cpu_parameters = {'criterion': ('gini', 'entropy', 'log_loss'),
                  'max_depth': [2, 4, 8], 'max_features': ('sqrt', 'log2')}
    gpu_parameters = {'split_criterion': ('gini', 'entropy', 'poisson'),
                  'max_depth': [2, 4, 8], 'max_features': ('sqrt', 'log2')}
    # all parameters combinations:
    all_combination_cpu_parameters = list(itertools.product(*cpu_parameters.values()))
    all_combination_gpu_parameters = list(itertools.product(*gpu_parameters.values()))

    for i in range(len(all_combination_cpu_parameters)):
        # for random forest
        cpu_cur_criterion = all_combination_cpu_parameters[i][0]
        cpu_cur_max_depth = all_combination_cpu_parameters[i][1]
        cpu_cur_max_features = all_combination_cpu_parameters[i][2]
        clf_rf = RandomForestClassifier(criterion=cpu_cur_criterion, max_depth=cpu_cur_max_depth, max_features=cpu_cur_max_features)
        sklearn_time_rf = timeit.timeit(lambda: train_data(clf_rf, X, y), number=5)

        gpu_cur_criterion = all_combination_gpu_parameters[i][0]
        gpu_cur_max_depth = all_combination_gpu_parameters[i][1]
        gpu_cur_max_features = all_combination_gpu_parameters[i][2]
        clf_rf = RandomForestClassifier_gpu(split_criterion=gpu_cur_criterion, max_depth=gpu_cur_max_depth, max_features=gpu_cur_max_features)
        cuml_time_rf = timeit.timeit(lambda: train_data(clf_rf, X, y), number=5)

        ratio_rf = sklearn_time_rf/cuml_time_rf

        print("Current cpu parameters - criterion: " + cpu_cur_criterion + " max_depth: " + str(cpu_cur_max_depth) + " max_features: " + cpu_cur_max_features)
        print(f"""(rf) Average time of sklearn's {clf_rf.__class__.__name__}""", sklearn_time_rf, 's')
        print("Current gpu parameters - split_criterion: " + str(gpu_cur_criterion) + " max_depth: " + str(gpu_cur_max_depth) + " max_features: " + gpu_cur_max_features)
        print(f"""(rf) Average time of cuml's {clf_rf.__class__.__name__}""", cuml_time_rf, 's')
        print('(rf) Ratio between sklearn and cuml is', ratio_rf)

        if num_rounds not in CPU_time_list:
            # initialize a list in a given dictionary key
            CPU_time_list[num_rounds] = {}
            GPU_time_list[num_rounds] = {}
            ratio_time_list[num_rounds] = {}
            CPU_time_list[num_rounds][(size_number, (cpu_cur_criterion, cpu_cur_max_depth, cpu_cur_max_features))] = (sklearn_time_rf)
            GPU_time_list[num_rounds][(size_number, (cpu_cur_criterion, cpu_cur_max_depth, cpu_cur_max_features))] = (cuml_time_rf)
            ratio_time_list[num_rounds][(size_number, (cpu_cur_criterion, cpu_cur_max_depth, cpu_cur_max_features))] = (ratio_rf)
        else:
            # append to an existing list
            CPU_time_list[num_rounds][(size_number, (gpu_cur_criterion, gpu_cur_max_depth, gpu_cur_max_features))] = (sklearn_time_rf)
            GPU_time_list[num_rounds][(size_number, (gpu_cur_criterion, gpu_cur_max_depth, gpu_cur_max_features))] = (cuml_time_rf)
            ratio_time_list[num_rounds][(size_number, (gpu_cur_criterion, gpu_cur_max_depth, gpu_cur_max_features))] = (ratio_rf)

def run_nb(num_rounds, size_number, CPU_time_list, GPU_time_list, ratio_time_list, X, y):
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

def run_boosting(num_rounds, size_number, CPU_time_list, GPU_time_list, ratio_time_list, X, y):
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
    # do Grid Search for boosting
    parameters = {'n_estimators': [10, 50, 100], 'learning_rate': [0.1, 0.5, 1.0]}
    # all parameters combinations:
    all_combination_parameters = itertools.product(*parameters.values())

    for a_combination_parameter in all_combination_parameters:
        # for boosting
        cur_n_estimators = a_combination_parameter[0]
        cur_learning_rate = a_combination_parameter[1]
        clf_ba = AdaBoostClassifier(n_estimators=cur_n_estimators, learning_rate=cur_learning_rate)
        sklearn_time_ba = timeit.timeit(lambda: train_data(clf_ba, X, y), number=5)

        clf_ba = LGBMClassifier(n_estimators=cur_n_estimators, learning_rate=cur_learning_rate)
        cuml_time_ba = timeit.timeit(lambda: train_data(clf_ba, X, y), number=5)

        ratio_ba = sklearn_time_ba/cuml_time_ba

        print("Current parameters - n_estimator: " + str(cur_n_estimators) + " learning_rate: " + str(cur_learning_rate))
        print(f"""(ba) Average time of sklearn's {clf_ba.__class__.__name__}""", sklearn_time_ba, 's')
        print(f"""(ba) Average time of cuml's {clf_ba.__class__.__name__}""", cuml_time_ba, 's')
        print('(ba) Ratio between sklearn and cuml is', ratio_ba)

        if num_rounds not in CPU_time_list:
            # initialize a list in a given dictionary key
            CPU_time_list[num_rounds] = {}
            GPU_time_list[num_rounds] = {}
            ratio_time_list[num_rounds] = {}
            CPU_time_list[num_rounds][(size_number, (cur_n_estimators, cur_learning_rate))] = (sklearn_time_ba)
            GPU_time_list[num_rounds][(size_number, (cur_n_estimators, cur_learning_rate))] = (cuml_time_ba)
            ratio_time_list[num_rounds][(size_number, (cur_n_estimators, cur_learning_rate))] = (ratio_ba)
        else:
            # append to an existing list
            CPU_time_list[num_rounds][(size_number, (cur_n_estimators, cur_learning_rate))] = (sklearn_time_ba)
            GPU_time_list[num_rounds][(size_number, (cur_n_estimators, cur_learning_rate))] = (cuml_time_ba)
            ratio_time_list[num_rounds][(size_number, (cur_n_estimators, cur_learning_rate))] = (ratio_ba)

def plot_fiture(repeat_time, max_size, current_graph_name, sample_or_feature, directory_to_store, x_axis, CPU_time_list, GPU_time_list, ratio_time_list):
    # *** function parameters *** #
    # current_graph_name - indicate which learning model is used
    # sample_or_feature - mainly changing sample or feature
    # directory_to_store - directory name to store the graph
    # CPU_time_list - the list to store the CPU runtime
    # GPU_time_list - the list to store the GPU runtime
    # ratio_time_list - the list to store the runtime ratio between CPU/GPU
    # *** ------------------- *** #
    # cpu and gpu runtime plot
    figure(figsize=(10, 10), dpi=80)

    color_array = ["#F39C12", "#27AE60", "#2980B9", "#8E44AD", "#C0392B"] # array for color codinng
    s_or_f = "" # string for reverse name (between sample and feature)
    processed_CPU_time_list = {}
    processed_GPU_time_list = {}
    processed_ratio_time_list = {}
    # extract information to CPU_time_list
    if current_graph_name != "nb":
        for j in range(repeat_time):
            if j not in processed_CPU_time_list:
                processed_CPU_time_list[j] = {}
                processed_GPU_time_list[j] = {}
                processed_ratio_time_list[j] = {}
            for i in range(500, max_size+1, 500):
                processed_CPU_time_list[j][i] = [value for key,value in CPU_time_list[j].items() if key[0] == i]
                processed_GPU_time_list[j][i] = [value for key,value in GPU_time_list[j].items() if key[0] == i]
                processed_ratio_time_list[j][i] = [value for key,value in ratio_time_list[j].items() if key[0] == i]

    if sample_or_feature == "Sample": # find the reverse name (between sample and feature)
        s_or_f = "features"
    else:
        s_or_f = "samples"

    # plot multiple lines
    for i in range(repeat_time):
        num = 50 * (i+1)
        num = str(num)
        if current_graph_name != "nb":
            cpu_y_axis = [mean(value) for key,value in processed_CPU_time_list[i].items()]
            gpu_y_axis = [mean(value) for key,value in processed_GPU_time_list[i].items()]
            error_cpu_y_axis = [pstdev(value) for key,value in processed_CPU_time_list[i].items()]
            error_gpu_y_axis = [pstdev(value) for key,value in processed_GPU_time_list[i].items()]
            plt.plot(x_axis, cpu_y_axis, label="CPU " + num + " " + s_or_f, color=color_array[i], linestyle="--")
            plt.plot(x_axis, gpu_y_axis, label="GPU " + num + " " + s_or_f, color=color_array[i])
            plt.errorbar(x_axis, cpu_y_axis, yerr=error_cpu_y_axis, fmt=".", color=color_array[i])
            plt.errorbar(x_axis, gpu_y_axis, yerr=error_gpu_y_axis, fmt=".", color=color_array[i])
        else:
            plt.plot(x_axis, CPU_time_list[i], label="CPU " + num + " " + s_or_f, color=color_array[i], linestyle="--")
            plt.plot(x_axis, GPU_time_list[i], label="GPU " + num + " " + s_or_f, color=color_array[i])
    plt.legend(loc='upper right')
    if current_graph_name == "ba":
        plt.title("MSAP (CPU) classifier: Adaboost vs. (GPU) classifier: LightGBM")
    else:
        plt.title("MSAP (CPU) classifier: " + current_graph_name + " from sklearn vs. (GPU) classifier: " + current_graph_name  + " from cuml")
    plt.xlabel(sample_or_feature + ' Size')
    plt.ylabel('Computation Time(s)')
    plt.savefig(directory_to_store + current_graph_name + '_compare_' + sample_or_feature +'_plot.png')
    plt.cla()

    # cpu/gpu ratio plot
    figure(figsize=(10, 10), dpi=80)

    for i in range(repeat_time):
        num = 50 * (i+1)
        num = str(num)
        if current_graph_name != "nb":
            ratio_y_axis = [cpu_y_axis[j]/gpu_y_axis[j] for j in range(len(cpu_y_axis))]
            plt.plot(x_axis, ratio_y_axis, label="50 features", color=color_array[i])
        else:
            plt.plot(x_axis, ratio_time_list[i], label="50 features", color=color_array[i])
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
    s_svc_average_time_sklearn_all = {}
    s_svc_average_time_cuml_all = {}
    s_svc_ratio_between_all = {}
    s_rf_average_time_sklearn_all = {}
    s_rf_average_time_cuml_all = {}
    s_rf_ratio_between_all = {}
    s_nb_average_time_sklearn_all = [[], [], [], [], []]
    s_nb_average_time_cuml_all = [[], [], [], [], []]
    s_nb_ratio_between_all = [[], [], [], [], []]
    s_ba_average_time_cpu_all = {}
    s_ba_average_time_gpu_all = {}
    s_ba_ratio_between_all = {}
    # total repeat time while increase sample/feature size
    repeat_time = 2
    # maximum size for sample/feature
    max_size = 1000
    num_features = 50 # initial number of features

    # five rounds, increase feature numbers (50, 100, 150, 200, 250, 300)
    for f_index in range(repeat_time):
        sample_numbers = 0 # reinitialize sample size
        num_features = 50 * (f_index + 1) # incrase feature number
        
        while sample_numbers < max_size:
            sample_numbers += 500 # update sample size
            print("(change sample size: ", sample_numbers, num_features,")") # print  indication for rounds
            X, y  = datasets.make_classification(n_samples=sample_numbers, n_features = num_features) # generate sample data

            # some cuML models require the input to be np.float32
            X = X.astype(np.float32)
            y = y.astype(np.float32)

            # *** train different learning model and record runtime *** #
            # for svc
            run_svc(f_index, sample_numbers, s_svc_average_time_sklearn_all, s_svc_average_time_cuml_all, s_svc_ratio_between_all, X, y)
            # for random forest
            run_rf(f_index, sample_numbers, s_rf_average_time_sklearn_all, s_rf_average_time_cuml_all, s_rf_ratio_between_all, X, y)
            # for nb
            run_nb(f_index, sample_numbers, s_nb_average_time_sklearn_all, s_nb_average_time_cuml_all, s_nb_ratio_between_all, X, y)
            # for boosting
            run_boosting(f_index, sample_numbers, s_ba_average_time_cpu_all, s_ba_average_time_gpu_all, s_ba_ratio_between_all, X, y)

    # define x-axis for plotting
    x_axis = [ i for i in range(500, max_size+1, 500)]

    # *** generate plots for changing sample size *** #
    # plot svc
    plot_fiture(repeat_time, max_size, "svc", "Sample", "./update_sample/", x_axis, s_svc_average_time_sklearn_all, s_svc_average_time_cuml_all, s_svc_ratio_between_all)
    # plot rf
    plot_fiture(repeat_time, max_size, "rf", "Sample", "./update_sample/", x_axis, s_rf_average_time_sklearn_all, s_rf_average_time_cuml_all, s_rf_ratio_between_all)
    # plot nb
    plot_fiture(repeat_time, max_size, "nb", "Sample", "./update_sample/", x_axis, s_nb_average_time_sklearn_all, s_nb_average_time_cuml_all, s_nb_ratio_between_all)
    # plot ba
    plot_fiture(repeat_time, max_size, "ba", "Sample", "./update_sample/", x_axis, s_ba_average_time_cpu_all, s_ba_average_time_gpu_all, s_ba_ratio_between_all)

    ### Saving results ###
    final_result["change_sample_size"] = {}
    final_result["change_sample_size"]["svc"] = [copy.deepcopy(s_svc_average_time_sklearn_all), copy.deepcopy(s_svc_average_time_cuml_all), copy.deepcopy(s_svc_ratio_between_all)]
    final_result["change_sample_size"]["rf"] = [copy.deepcopy(s_rf_average_time_sklearn_all), copy.deepcopy(s_rf_average_time_cuml_all), copy.deepcopy(s_rf_ratio_between_all)]
    final_result["change_sample_size"]["nb"] = [copy.deepcopy(s_nb_average_time_sklearn_all), copy.deepcopy(s_nb_average_time_cuml_all), copy.deepcopy(s_nb_ratio_between_all)]
    final_result["change_sample_size"]["ba"] = [copy.deepcopy(s_ba_average_time_cpu_all), copy.deepcopy(s_ba_average_time_gpu_all), copy.deepcopy(s_ba_ratio_between_all)]


    #----------------------------change feature size----------------------------#
    # declare lists to store CPU and GPU runtimes
    f_svc_average_time_sklearn_all = {}
    f_svc_average_time_cuml_all = {}
    f_svc_ratio_between_all = {}
    f_rf_average_time_sklearn_all = {}
    f_rf_average_time_cuml_all = {}
    f_rf_ratio_between_all = {}
    f_nb_average_time_sklearn_all = [[], [], [], [], []]
    f_nb_average_time_cuml_all = [[], [], [], [], []]
    f_nb_ratio_between_all = [[], [], [], [], []]
    f_ba_average_time_cpu_all = {}
    f_ba_average_time_gpu_all = {}
    f_ba_ratio_between_all = {}
    num_samples = 50 # initial number of samples

    # five rounds, increase sample numbers (50, 100, 150, 200, 250, 300)
    for f_index in range(repeat_time):
        feature_numbers = 0 # reinitialize feature size
        num_samples = 50 * (f_index + 1) # incrase sample number

        while feature_numbers < max_size:
            feature_numbers += 500 # update feature size
            print("(change feature size: ", num_samples, feature_numbers,")")# print  indication for rounds
            X, y  = datasets.make_classification(n_samples=num_samples, n_features=feature_numbers) # generate sample data

            # some cuML models require the input to be np.float32
            X = X.astype(np.float32)
            y = y.astype(np.float32)

            # *** train different learning model and record runtime *** #
            # for svc
            run_svc(f_index, feature_numbers, f_svc_average_time_sklearn_all, f_svc_average_time_cuml_all, f_svc_ratio_between_all, X, y)
            # for random forest
            run_rf(f_index, feature_numbers, f_rf_average_time_sklearn_all, f_rf_average_time_cuml_all, f_rf_ratio_between_all, X, y)
            # for nb
            run_nb(f_index, feature_numbers, f_nb_average_time_sklearn_all, f_nb_average_time_cuml_all, f_nb_ratio_between_all, X, y)
            # for boosting
            run_boosting(f_index, feature_numbers, f_ba_average_time_cpu_all, f_ba_average_time_gpu_all, f_ba_ratio_between_all, X, y)
    
    # define x-axis for plotting
    # this is already defined before

    # *** generate plots for changing sample size *** #
    # plot svc
    plot_fiture(repeat_time, max_size, "svc", "Feature", "./update_feature/", x_axis, f_svc_average_time_sklearn_all, f_svc_average_time_cuml_all, f_svc_ratio_between_all)
    # plot rf
    plot_fiture(repeat_time, max_size, "rf", "Feature", "./update_feature/", x_axis, f_rf_average_time_sklearn_all, f_rf_average_time_cuml_all, f_rf_ratio_between_all)
    # plot nb
    plot_fiture(repeat_time, max_size, "nb", "Feature", "./update_feature/", x_axis, f_nb_average_time_sklearn_all, f_nb_average_time_cuml_all, f_nb_ratio_between_all)
    # plot ba
    plot_fiture(repeat_time, max_size, "ba", "Feature", "./update_feature/", x_axis, f_ba_average_time_cpu_all, f_ba_average_time_gpu_all, f_ba_ratio_between_all)

    ### Saving results ###
    final_result["change_feature_size"] = {}
    final_result["change_feature_size"]["svc"] = [copy.deepcopy(f_svc_average_time_sklearn_all), copy.deepcopy(f_svc_average_time_cuml_all), copy.deepcopy(f_svc_ratio_between_all)]
    final_result["change_feature_size"]["rf"] = [copy.deepcopy(f_rf_average_time_sklearn_all), copy.deepcopy(f_rf_average_time_cuml_all), copy.deepcopy(f_rf_ratio_between_all)]
    final_result["change_feature_size"]["nb"] = [copy.deepcopy(f_nb_average_time_sklearn_all), copy.deepcopy(f_nb_average_time_cuml_all), copy.deepcopy(f_nb_ratio_between_all)]
    final_result["change_feature_size"]["ba"] = [copy.deepcopy(f_ba_average_time_cpu_all), copy.deepcopy(f_ba_average_time_gpu_all), copy.deepcopy(f_ba_ratio_between_all)]


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