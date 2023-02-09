############################import needed function############################
from _learning_models import run_boosting
from _learning_models import plot_figure


############################main function############################
if __name__ == '__main__':

    # store all the results in a dictionary
    final_result = {}

    # total repeat time while increase sample/feature size
    repeat_time = 5
    # maximum size for sample/feature
    max_size = 5000

    #----------------------------change feature size----------------------------#
    # declare lists to store CPU and GPU runtimes
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
            # for boosting
            run_boosting(f_index, feature_numbers, f_ba_average_time_cpu_all, f_ba_average_time_gpu_all, f_ba_ratio_between_all, X, y)
    
    # define x-axis for plotting
    x_axis = [ i for i in range(500, max_size+1, 500)]

    # *** generate plots for changing sample size *** #
    # plot ba
    plot_figure(repeat_time, max_size, "ba", "Feature", "./update_feature/", x_axis, f_ba_average_time_cpu_all, f_ba_average_time_gpu_all, f_ba_ratio_between_all)

    ### Saving results ###
    final_result["change_feature_size"] = {}
    final_result["change_feature_size"]["ba"] = [copy.deepcopy(f_ba_average_time_cpu_all), copy.deepcopy(f_ba_average_time_gpu_all), copy.deepcopy(f_ba_ratio_between_all)]


    # dump output to a pickle file
    with open('./outputs/comparison_plot_results_feature_ba.pickle', 'wb') as handle:
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