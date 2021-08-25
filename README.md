# Bio-inspired intelligence and learning for aerospace applications

This is the git repository used for the assignment of the course AE4350 Bio-Inspired Intelligence and Learning for Aerospace Applications from the faculty of Aerospace Engineering at Delft University of Technology.
It follows an implementation of the DDPG paper by DeepMind, using the code from Dr. Phil Tabor ([link](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/DDPG/pytorch/lunar-lander)), along with some alterations I made myself for this assignment.
Below is an overview of the code structure, how it works and in short what I changed from mr. Tabor's code.

## Files
* config.json: This file contains all the hyperparameters governing the neural network architecture and training. This is changed by the code for the main experiment and hyperparameter check.
* ddpg.py: This file contains the main network classes for the implementation of the DDPG algorithm.
* LunarLanderMain.py: This file contains the instantiation of the agent as defined in ddpg.py and creates the 'LunarLanderContinuous-v2' environment, cycles through the episodes and through the timesteps within the episode and trains the agent on the environment.
* main.py: This file changes the config file and calls LunarLanderMain for each alteration of the config file to carry out the main experiment.
* visual_inspection.py: Running this file loads a certain pretrained architecture (specify in the script) and shows the rendered version for visual inspection of the solution.
* validation.py: This file changes the config file and calls LunarLanderMain for the validation experiment.
* hyperparam_check.py: This file changes the config file hyperparameters and calls LunarLanderMain for each hyperparameter iteration for the sensitivity analysis (hyperparameter check).
* utils.py: Contains some utility functions such as saving and plotting.


## Folders
* original_code: Contains original code by Dr. Tabor with only a couple of lines changed to carry out the validation experiment. Results from this code are in this folder as well.
* results_val: Contains the results generated by validation.py
* results_main0, results_main1, results_main2: Contain the results for the different architecture settings as created by main.py
* results_hypercheck: Contains results for the hyperparameter check generated by hyperparam_check.py

## How to use
Make sure to have installed all dependencies, installing pybullet and gym may require you to install [SWIG](http://www.swig.org/).
To reproduce the results presented in the report (PDF in the repository), simply run main.py with the preset hyperparameters. 

Further, control of the architecture and other hyperparameters mainly work through the config file, so changing the architecture (defined as a list of the number of nodes per hidden layer, so [10, 80, 3] will create 3 hidden layers with 10 nodes in layer 1, 80 nodes in layer 2 and 3 nodes in layer 3) can be done via the config. Input and output layers are defined by the environment.

If you want to train only one time with no variation to hyperparameters or network architecture, set these in the config file and run the LunarLanderMain.py file.

Be aware as results may be overwritten, results in the github repository are the results from the run as described in the report. Project was done on Windows and may require minimal modification to work on Unix-based machines.
