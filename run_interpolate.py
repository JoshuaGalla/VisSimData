import sys
import json
import numpy as np
from analysis.config import config
from analysis.interpolate import fit_gp
from analysis.plot import sim_data_plots
from analysis.sim_data import sim_data_analysis

# Load in parameters
config = config(file='parameters/parameters.yaml')

neuron = config.neuron

x_star, stim_data, response_data, avg_map, param1, param2, neuron = sim_data_analysis().get_sim_data(neuron)

f, sigma = fit_gp().calc_interp(stim_data, response_data, x_star, config.var, config.gamma, config.eta)
print(f)
print(sigma)

with open('./data/interp_data_f.npy', 'wb') as file:
    np.save(file,f)
with open('./data/interp_data_sigma.npy', 'wb') as file:
    np.save(file,sigma)

sim_data_plots().plot_tc(avg_map, param1, param2, neuron)
sim_data_plots().plot_tc_interp(param1, param2, neuron)

sys.exit()