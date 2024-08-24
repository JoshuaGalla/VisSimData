import yaml
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

class sim_data_analysis(): #carries out analysis, exclusion, and correction of neural data
    
    def __init__(self):     
        
        #generating data
        self.num_stimuli = None
        
        #retrieving neural responses
        self.neurons_responses_corrected = None 
        self.avg_responses_dict = None 
        self.savgol_avg = None
        
        #matching response to stim
        self.all_stim_info = None
        
        #retrieving tuning curve info
        self.all_combos = None 
        self.avg_map = None 
        self.param1 = None 
        self.param2 = None
            
    def read_config(self, file):
        
        """
        reads in response_data and stim_data

        Args:
            file (str): .yml file of parameters ("parameters.yml")

        Returns:
            None
        """
        
        with open(file, 'r') as file:
            parameters = yaml.safe_load(file)
        
        self.response_data = parameters['General']['response_data']
        self.stim_data = parameters['General']['stim_data']
        self.seed = parameters['General']['seed']
        
        self.stim_interval = parameters['Simulate']['stim_interval']
        
        self.param1 = list(range(parameters['Simulate']['dim1_min'], parameters['Simulate']['dim1_max']+1))
        self.param2 = list(range(parameters['Simulate']['dim2_min'], parameters['Simulate']['dim2_max']+1))
        
        self.num_neurons = parameters['Neurons']['num_neurons']
        self.num_frames = parameters['Simulate']['num_frames']  
        self.max_response = parameters['Simulate']['max_response'] 
        self.no_response_prob = parameters['Simulate']['no_response_prob']  
                
    def calc_exp(self, neuron):
        
        """
        carries out respective function for calculating avg neural response, applying savitzgy-golay filter, and stimulus-specific 
        responses

        Args:
            None

        Returns:
            None
        """
        
        print("Generating stim data...")
        self.num_stimuli = self.generate_stim_data()
        
        print("Generating response data...")
        self.generate_stim_responses()
        
        print("Retreiving responses...")
        self.neurons_responses_corrected, self.avg_responses_dict, self.savgol_avg = self.get_responses()
        
        print("Matching responses to stim characteristics...")
        self.all_stim_info = self.stim_combo_analysis()
        
        print("Defining tuning curve characteristics...")
        self.all_combos, self.avg_map = self.tuning_curves_analysis(neuron)
    
    def generate_stim_data(self):
    
        """
        Generates text file containing frame number and characteristics of visual stim occurrences

        Args:
            None

        Returns:
            num_stimuli (int): Number of stimuli occurrences in text file
        """

        np.random.seed(self.seed)
        #make values for stim onset
        col1 = np.arange(self.stim_interval, self.num_frames, self.stim_interval)

        num_stimuli = len(col1)

        #create random values for dim1 and dim2
        col2 = np.random.randint(min(self.param1), max(self.param1)+1, size=num_stimuli)
        col3 = np.random.randint(min(self.param1), max(self.param2)+1, size=num_stimuli)  

        #combine
        data = np.column_stack((col1, col2, col3))

        #save
        file_name = "./data/stim_data.txt"
        np.savetxt(file_name, data, delimiter=' ', fmt='%d')
        num_stimuli = (self.num_frames // self.stim_interval)

        print(f"File {file_name} has been created with {num_stimuli-1} stimuli")
    
        return num_stimuli 

    def generate_ap(self, width, amplitude):
    
        """
        Generates artifical action potential (ap) after stimulus occurrence

        Args:
            width (int): random integer between 5 and 50 representing width of ap
            amplitude (float): random float between 1 and max_response representing peak

        Returns:
            ap (float): value at given location along width of action potential for respective stim occurrence
        """

        ap = np.zeros(width)
        midpoint = width // 2

        #make Gaussian-like spike
        for i in range(width):
            ap[i] = amplitude * np.exp(-((i - midpoint) ** 2) / (2 * (width / 6) ** 2))

        return ap

    def generate_stim_responses(self):

        """
        Generates each neuron's responses and outputs text file

        Args:
            None

        Returns:
            None, but generates stim_responses.txt file
        """

        np.random.seed(self.seed)
        #initialize matrix with zeros
        matrix = np.zeros((self.num_neurons, self.num_frames))

        #make aps and insert into matrix
        for i in range(1, self.num_stimuli):
            stimulus_start = i * self.stim_interval
            stimulus_end = stimulus_start + self.stim_interval

            for neuron in range(self.num_neurons):

                if np.random.rand() < self.no_response_prob:
                    #no response for this stimulus
                    continue

                #create random amplitude and width for each neuron
                amplitude = np.random.uniform(1, self.max_response) 
                width = np.random.randint(5, 50)
                ap = self.generate_ap(width, amplitude)

                #place ap into matrix
                ap_start = stimulus_start + np.random.randint(0, self.stim_interval - width)
                ap_end = ap_start + width
                matrix[neuron, ap_start:ap_end] += ap

        file_name = "./data/stim_responses.txt"
        np.savetxt(file_name, matrix, fmt='%.2f')

        print(f"File {file_name} has been created for {self.num_neurons} neurons with {self.num_frames} responses")
    
    def get_responses(self):

        """
        Retrieves responses at each frame for each neuron, calculates averages, and corrects 'background noise'

        Args:
            None

        Returns:
            neuron_responses_corrected (dict): dict of all neurons and their corrected responses
            avg_responses_dict (dict): dict of the avg response of each frame across all neurons
            savgol_avg (array): Fitted smoothed line using Savitzky-Golay filter representing average to correct for noise
        """

        with open('./data' + self.response_data, 'r') as file:
            neurons_responses = file.readlines()

        neurons_responses_dict = {key:[] for key in range(1, (len(neurons_responses)+1))}

        #append response at each frame for each neuron
        for neuron in range(len(neurons_responses)):
            neurons_responses_dict[neuron+1] = neurons_responses[neuron].strip("\n").split(" ")

        for neuron, responses in neurons_responses_dict.items():
            neurons_responses_dict[neuron] = [float(response) for response in responses if response != 'nan']

        responses_list = list(iter(neurons_responses_dict.values()))

        avg_responses_dict={'N_avg':[] for value in range(self.num_frames)}

        #get frame averages column-wise
        for response in range(len(responses_list[0])):
            frame_list = [frame[response] for frame in responses_list] 
            avg_responses_dict['N_avg'].append(sum(frame_list)/len(frame_list)) 

        #apply savitzky-golay smoothed filter to avg responses  
        avg_responses_list = list(iter(avg_responses_dict.values()))
        avg_responses_array = np.array(avg_responses_list)
        savgol_avg = savgol_filter(avg_responses_array, 300, 3) 

        neurons_responses_corrected = {}

        #correct each neuron by dividing response at each frame by savgol-fitted avg response at each frame
        for neuron, responses in neurons_responses_dict.items():
            corrected_responses = []
            for frame, response in enumerate(responses):
                corrected_responses.append(response / savgol_avg[0][frame])

            neurons_responses_corrected[neuron] = corrected_responses

        return neurons_responses_corrected, avg_responses_dict, savgol_avg, 
    
    def stim_combo_analysis(self):
    
        """
        Retrieves stimulus info from stim_data.txt file

        Args:
            None

        Returns:
            all_stim_info (list): list containing characteristics for each stimulus occurrence 
        """

        with open('./data' + self.stim_data) as file:
            stimuli = file.readlines()

        all_stim_info = []
        for i in zip(stimuli):
            all_stim_info.append(i[0].strip("\n").split(" "))

        #convert stim to floats and append to list
        for idx, stim in enumerate(all_stim_info):
            stim = [eval(param) for param in stim]
            all_stim_info[idx] = stim

        return all_stim_info
    
    def tuning_curves_analysis(self, neuron):
    
        """
        Calculates average response for specified neuron(s) and maps response to stimulus characteristics

        Args:
            neuron (int/str): neuron to be plotted (int between 1:num_neurons), or average of all neurons ('all')

        Returns:
            all_combos (array): array of every unique combo between possibly tested parameters
            avg_map (dict): average of responses associated with a combo of tested parameters
        """

        all_combos = np.array([(x, y) for x in self.param1 for y in self.param2])

        param1_frames = {key:[] for key in self.param1}
        param2_frames = {key:[] for key in self.param2}

        #appending frame of matching unqiue param tested as value
        for stim in self.all_stim_info:
            stim_param1 = stim[1]
            stim_param2 = stim[2]

            for param in self.param1:
                if param == stim_param1:
                    param1_frames[param].append(stim[0])

            for param in self.param2:
                if param == stim_param2:
                    param2_frames[param].append(stim[0])

        #dict that will contain stimulus onset frame and the neural responses within the stimulus window
        stim_responses = {}

        #assign the values (stimulus onset frames) from parameter_uniqvals_frames_dict their own dict as a key
        for param, frames in param1_frames.items():
            for frame in frames:
                stim_responses[frame] = []

        for frame, response in stim_responses.items():
            frame_idx = 0
            for stim in self.all_stim_info:
                frame_idx += 1

                #finding range/window of frames that correspond to stimulus onset
                if frame == stim[0] and frame_idx < len(self.all_stim_info):
                    frame_start = int(frame)
                    frame_end = int(self.all_stim_info[frame_idx][0])

                    #adding neural response frames that correspond to selected stimulus window
                    if neuron == 'all':
                        for avg_frame in self.avg_responses_dict['N_avg'][frame_start:frame_end]:
                            stim_responses[frame].append(avg_frame)

                    else:
                        for neuron_frame in self.neurons_responses_corrected[neuron][frame_start:frame_end]:
                            stim_responses[frame].append(neuron_frame)

                #identifying frame window when analyzing last stimulus onset
                elif frame == stim[0] and frame_idx == len(self.all_stim_info):
                    frame_start = int(frame)
                    frame_end = int(len(self.avg_responses_dict['N_avg']))

                    #adding neural response frames that correspond to selected stimulus window
                    if neuron == 'all':
                        for avg_frame in self.avg_responses_dict['N_avg'][frame_start:frame_end]:
                            stim_responses[frame].append(avg_frame)

                    else:
                        for neuron_frame in self.neurons_responses_corrected[neuron][frame_start:frame_end]:
                            stim_responses[frame].append(neuron_frame)

        #calculating average of all responses within selected window for each different stimulus onset
        for stim, responses in stim_responses.items():
            if len(responses) == 0:
                stim_responses[stim] = np.nan
            else:
                stim_responses[stim] = (sum(responses)/len(responses))

        stim_responses = sorted(stim_responses.items())

        for idx, i in enumerate(stim_responses):
            self.all_stim_info[idx].append(i[1])

        #responses_map = {}
        avg_map = {}
        for row in all_combos:
            #responses_map[tuple(row)] = []
            avg_map[tuple(row)] = []

        #appending average values associated with unique combo of parameters
        for combo in avg_map:
            for stim in self.all_stim_info:
                if combo[0] == stim[1] and combo[1] == stim[2]:
                    #responses_map[combo].append(stim[3])
                    avg_map[combo].append(stim[3])

        #calculating average and replacing non-sampled spaces with nan
        for combo, response in avg_map.items():
            if len(response) > 0:
                avg_map[combo] = sum(response)/len(response)
                if avg_map[combo] == 0:
                    avg_map[combo] = np.nan
            else:
                avg_map[combo] = np.nan

        return all_combos, avg_map

    def get_sim_data(self, neuron):
        
        """
        retrieves x_star, stim_data, and response_data that serves as input for tuning curve interpolation

        Args:
            neuron (int): specific neuron to be analyzed

        Returns:
            x_star (array): array of shape (# of all unique stimuli combos, # of parameters)
            stim_data (array): array of shape (# of total stimuli shown in experiment, # of parameters)
            resonse_data (array): array of shape (# of neurons being analyzed, # of total stimuli shown in experiment)
        """
        
        self.read_config('parameters/parameters.yaml')
        self.calc_exp(neuron)
        
        x_star = self.all_combos
        
        stim_data = []
        response_data = []
        for stim in self.all_stim_info:
            stim_data.append(stim[1:3])
            response_data.append(stim[3])

        stim_data = np.array(stim_data)
        response_data = np.array(response_data)
        response_data = response_data.reshape(1,-1)
        
        avg_map = self.avg_map
        param1 = self.param1
        param2 = self.param2
                
        return x_star, stim_data, response_data, avg_map, param1, param2, neuron