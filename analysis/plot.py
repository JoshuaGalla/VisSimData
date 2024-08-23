import numpy as np
import matplotlib.pyplot as plt

class sim_data_plots(): #carries out plotting for experimental data
    
    def read_config(self, file):
        
        """
        reads in dataset, raw data, and stimulus data

        Args:
            file (str): .yml file of parameters ("parameters.yml")

        Returns:
            None
        """
        
        with open(file, 'r') as file:
            parameters = yaml.safe_load(file)
        
        self.dataset = parameters['General']['dataset']
        
        self.param1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.param2 = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
        
    def exp_data_plot(self, avg_map, neuron):
        
        """
        plots interpolated offline fit of tuning curve for selected neuron

        Args:
            neuron (int): specific neuron to be analyzed (ints 0-398 can be chosen)

        Returns:
            None, but displays interpolated tuning curve for selected neuron
        """
        
        self.read_config('parameters/zebrafish.yaml')
         
        with open(self.dataset + '/spots_data_f.npy', 'rb') as f:
            pred_means = np.load(f)
        with open(self.dataset + '/spots_data_sigma.npy', 'rb') as f:
            cov = np.load(f)
        
        known_values = np.array([response for response in avg_map.values() if response])
        interped_data = known_values.reshape(len(self.param1), len(self.param2))
        
        means = np.reshape(pred_means,(1,len(self.param1),len(self.param2)))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.imshow(interped_data, origin = 'lower', aspect = 'auto')
        plt.colorbar(label='Response')
        plt.xlabel('Speed (Velocity)')
        plt.ylabel('Size (Circle Radius)')
        plt.xticks(np.arange(len(self.param2)), self.param2)
        plt.yticks(np.arange(len(self.param1)), self.param1)
        plt.title(f'N{neuron} offline tuning curve input')
        plt.savefig(self.dataset + f'/N{neuron}_offline_input.png')
        plt.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.imshow(means[0], origin='lower', aspect = 'auto')
        plt.colorbar(label='Response')
        plt.xlabel('Speed (Velocity)')
        plt.ylabel('Size (Circle Radius)')
        plt.title(f'N{neuron} offline tuning curve output')
        plt.xticks(np.arange(len(self.param2)), self.param2)
        plt.yticks(np.arange(len(self.param1)), self.param1)
        plt.savefig(self.dataset + f'/N{neuron}_offline_output.png')
        plt.tight_layout()
        plt.show()
 
