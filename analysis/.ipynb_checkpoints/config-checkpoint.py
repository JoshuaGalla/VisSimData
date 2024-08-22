import yaml
import numpy as np
import pandas as pd

class Config():
    def __init__(self, file):
        self.params = self.read_config(file)
        self.__dict__.update(self.params)

    def read_config(self, file):
        
        """
        reads in dataset, parameters, define kernel and length, and initialize dimensions of data

        Args:
            file (str): .yml file of parameters ("parameters.yml")

        Returns:
            parameters (dict): dictionary of parameters from parameters.yml file that have been updated
        """
        
        with open(file, 'r') as file:
            parameters = yaml.safe_load(file)

        ## set parameters from 'General'
        np.random.seed(parameters['General']['seed'])
        self.method = parameters['General']['method']
        
        self.response_data = parameters['General']['response_data'] 
        self.stim_data = parameters['General']['stim_data'] 
        self.neuron = parameters['Neurons']['neuron'] 

        self.kernels = np.array([parameters['Interpolate']['kernel'], [parameters['Interpolate']['kernel']]) 

        exs = []
        xs = np.meshgrid(*exs)
        x_star = np.empty(xs[0].shape + (2,))
        for i in range(2):
            x_star[...,i] = xs[i]
        self.x_star = x_star.reshape(-1, 2)
        
        self.gamma = np.array([parameters['Interpolate']['gamma'], parameters['Interpolate']['gamma']])
        self.var = parameters['Interpolate']['var']                     
        self.nu = float(parameters['Interpolate']['nu'])                
        self.eta = float(parameters['Interpolate']['eta'])              

        return parameters
