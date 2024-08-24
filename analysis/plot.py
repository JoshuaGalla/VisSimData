import numpy as np
import matplotlib.pyplot as plt

class sim_data_plots(): #carries out plotting for tuning curves
        
    def plot_tc(self, avg_map, param1, param2, neuron):
        
        """
        plots non-interpolated tuning curve for selected neuron

        Args:
            neuron (int): specific neuron to be analyzed

        Returns:
            None, but displays non-interpolated tuning curve for selected neuron
        """
        
        known_values = np.array([response for response in avg_map.values() if response])
        interped_data = known_values.reshape(len(param1), len(param2))
                
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.imshow(interped_data.T, origin = 'lower', aspect = 'auto')
        plt.colorbar(label='Response')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.xticks(np.arange(len(param1)), param1)
        plt.yticks(np.arange(len(param2)), param2)
        if neuron == 'all':
            plt.title('Non-Interpolated Tuning Curve for Avg Neural Response')
        else:
            plt.title(f'N{neuron} Non-Interpolated Tuning Curve')
        plt.tight_layout()
        plt.show()
        
    def plot_tc_interp(self, param1, param2, neuron):

        """
        plots interpolated tuning curve for selected neuron

        Args:
            neuron (int): specific neuron to be analyzed

        Returns:
            None, but displays interpolated tuning curve for selected neuron
        """
 
        with open('./data/interp_data_f.npy', 'rb') as f:
            mean = np.load(f)
        with open('./data/interp_data_sigma.npy', 'rb') as sigma:
            cov = np.load(sigma)
            
        mean = np.reshape(mean,(1,len(param1),len(param2)))

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.imshow(mean[0].T, origin='lower', aspect = 'auto')
        plt.colorbar(label='Response')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        if neuron == 'all':
            plt.title('Interpolated Tuning Curve for Avg Neural Response')
        else:
            plt.title(f'N{neuron} Interpolated Tuning Curve')
        plt.xticks(np.arange(len(param1)), param1)
        plt.yticks(np.arange(len(param2)), param2)
        plt.tight_layout()
        plt.show()