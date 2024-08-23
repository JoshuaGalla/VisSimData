import sys
import json
import numpy as np
from analysis.config import config
from analysis.interpolate import calc_interp
from plots import plot_tc_sim, plot_tc_control, plot_interp_sim, plot_interp_control
from model.sim_data import sim_data_analysis

# Load in parameters
config = Config(file='parameters.yaml')

# Run 
#if len(sys.argv) > 1:
#    eval_method = sys.argv[1]
#else:
#    eval_method = None

            
if config.method == 'interp_sim': #offline tuning curve fit for exp data (individual neurons)

    response_data = config.response_data
    sim_data = config.sim_data
    
    neuron = config.neuron

    my_x_star, my_stim_data, my_response_data, avg_map, neuron = sim_data().get_sim_data(neuron)

    f, sigma = interpolate(stim_data, response_data, my_x_star, config.var, config.gamma, config.eta, config.kernels)

    with open(dataset + '/sim_data_f.npy', 'wb') as file:
        np.save(file,f)
    with open(dataset + '/sim_data_sigma.npy', 'wb') as file:
        np.save(file,sigma)

    spots_plots_exp().exp_data_plot(avg_map, neuron)

    sys.exit()
    
else:
    config.method == None

rsPr_list = random_sampling(config, print_flag=True)

pr_lists = {'Pr_list': Pr_list, 'rsPr_list': rsPr_list}
#file_path = f'/Users/anjshnkr/Desktop/DraelosLab/BayesOpt/docs/pr_lists_d={config.d}.txt'
#with open(file_path, 'w') as file:
    #json.dump(pr_lists, file)
    
plot_correct_prediction(N=config.N, Pr_list=Pr_list, rsPr_list=rsPr_list)
plot_peak_value(x1=config.exs[0], x2=config.exs[1], mse_final=mse_final, loc_list=loc_list, SimPop=config.SimPop)

# plot_mse(mse_final=mse_final)
# plot_mse_runtime_map(MSE=mse_final, runt_list=runt_list, vmin=min(mse_final), vmax=max(mse_final))
plot_run_time(test_time_neuron=test_time_neuron, average=True)
