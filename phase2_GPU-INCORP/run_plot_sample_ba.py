############################import useful libraries############################
from _learning_models import run_svc
from _learning_models import plot_figure


############################main function############################
if __name__ == '__main__':

    # store all the results in a dictionary
    final_result = {}

    #----------------------------change sample size----------------------------#
    # declare lists to store CPU and GPU runtimes
    s_ba_average_time_cpu_all = {}
    s_ba_average_time_gpu_all = {}
    s_ba_ratio_between_all = {}
    # total repeat time while increase sample/feature size
    repeat_time = 5
    # maximum size for sample/feature
    max_size = 5000
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
            # for boosting
            run_boosting(f_index, sample_numbers, s_ba_average_time_cpu_all, s_ba_average_time_gpu_all, s_ba_ratio_between_all, X, y)

    # define x-axis for plotting
    x_axis = [ i for i in range(500, max_size+1, 500)]

    # *** generate plots for changing sample size *** #
    # plot ba
    plot_fiture(repeat_time, max_size, "ba", "Sample", "./update_sample/", x_axis, s_ba_average_time_cpu_all, s_ba_average_time_gpu_all, s_ba_ratio_between_all)

    ### Saving results ###
    final_result["change_sample_size"] = {}
    final_result["change_sample_size"]["ba"] = [copy.deepcopy(s_ba_average_time_cpu_all), copy.deepcopy(s_ba_average_time_gpu_all), copy.deepcopy(s_ba_ratio_between_all)]

    # dump output to a pickle file
    with open('./outputs/comparison_plot_results_sample_ba.pickle', 'wb') as handle:
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