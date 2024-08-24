import yaml
import numpy as np
import pandas as pd

class config():
    def __init__(self, file):
        self.params = self.read_config(file)
        self.__dict__.update(self.params)

    def read_config(self, file):
        
        """
        reads in dataset, parameters, and initializes array dimensions of data

        Args:
            file (str): .yml file of parameters ("parameters.yml")

        Returns:
            None
        """
        
        with open(file, 'r') as file:
            parameters = yaml.safe_load(file)

        #set parameters from General
        np.random.seed(parameters['General']['seed'])
        self.response_data = parameters['General']['response_data'] 
        self.stim_data = parameters['General']['stim_data'] 

        #set parameters from Neurons
        self.num_neurons = parameters['Neurons']['num_neurons']
        self.neuron = parameters['Neurons']['neuron'] 

        #set array for interpolation
        exs = []
        for i in range(1, 3):
            exs.append(np.arange(10))

        xs = np.meshgrid(*exs)
        x_star = np.empty(xs[0].shape + (2,))
        for i in range(2):
            x_star[...,i] = xs[i]
        self.x_star = x_star.reshape(-1, 2)
        
        #set parameters from Interpolate
        self.gamma = np.array([parameters['Interpolate']['gamma'], parameters['Interpolate']['gamma']])
        self.var = parameters['Interpolate']['var']                     
        self.nu = float(parameters['Interpolate']['nu'])                
        self.eta = float(parameters['Interpolate']['eta'])   
        
        return parameters