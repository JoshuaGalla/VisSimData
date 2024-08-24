import math
import numpy as np
import scipy.stats as stats
import numba as nb
from analysis.kernel import kernel


class fit_gp():
    
    
    
    def calc_interp(self, stim_data, response_data, x_star, var, gamma, eta):
        f = np.empty((response_data.shape[0], x_star.shape[0]))
        sigma = np.empty((response_data.shape[0], x_star.shape[0], x_star.shape[0]))
        for neuron in range(response_data.shape[0]):
            X_t = stim_data
            y_neuron = response_data[neuron] #1D arr (T,)

            # Kernel matrices
            T = X_t.shape[0]
            K_t = kernel(X_t, X_t, var, gamma)
            k_star = kernel(X_t, x_star, var, gamma)
            kvv = kernel(x_star, x_star, var, gamma)

            # Compute GP parameters
            A = np.linalg.inv(K_t + eta**2 * np.eye(T))     # TO DO - look at pinv 
            f[neuron] = k_star.T @ A @ y_neuron                    # predicted means
            sigma[neuron] = var * np.eye(x_star.shape[0]) - k_star.T @ A @ k_star    # covariance matrix
            t = T

        return f, sigma
