# **Read Me:**

Repository for VisSimData.

# **Project Overview:**

The goal of this project is to perform a comprehensive analysis to understand how simulated visual stimuli parameters/characteristics influence neural response strength. This analysis consists of generating visualizations for both individual and population averages of simulated neurons, including neural traces, parameter-specific combo plots, sampling space heatmaps, and more.

# **Data:**

The project contained within this repository specifically applies visualization techniques to simulated visual stimulation data in theoretical biological model systems. This analysis can be viewed by opening the Jupyter Notebook titled "sim_data_analysis.ipynb". The following text files serve as input for this notebook:

1) "stim_responses.txt": contains an nxd array of values, where each row corresponds to a different simulated neuron, and each column is the respective neuron's response in time.
2) "stim_data.txt": contains three columns of data. The first column being the frame/time in which a simulated visual stimuli would appear, the second being the property of the first dimension of the visual stimulus shown, and the third being the property of the second dimension of the visual stimulus shown.

This analysis can help us identify a combination of properties for dimension 1 and 2 of a simulated visual stimulus that elicits the greatest response across our population of neurons. In an experimental context, this could help inform us of the characteristics of stimuli that elicit potential behavioral responses such as motor movement or pathways involved in visual processing.

# **Other Notes/References:**

A more complex and real-time version of this analysis is included as supplementary work in the BayesOpt repository of [the Draelos Lab](https://github.com/draeloslab) at University of Michigan. We apply statistical techniques including machine learning and Bayesian Optimization to high-dimensional visual stimulation experiments to estimate neural responses that inform us of behavioral dynamics and outputs. For more information on how this work can be applied to experimental data rather than simulated data, please visit the Draelos Lab repository linked above or our [lab website](https://draeloslab.org/).
