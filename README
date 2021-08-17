Implementation and ablation study of the DDPG Reinforcement Learning algorithm for OpenAI gym's LunarLanderContinuous-v2 environment.

This is the git repository used for the assignment of the course AE4350 Bio-Inspired Intelligence and Learning for Aerospace Applications from the faculty of Aerospace Engineering at Delft University of Technology.
It follows an implementation of the DDPG paper by DeepMind, using the code from Phil Tabor (link his repo), along with some alterations I made myself for this assignment.
Below is an overview of the code structure and in short what I changed from mr. Tabor's code.

config.json
This file contains all the hyperparameters governing the neural network architecture and training.

How to use
To reproduce the results presented in the report (PDF in the repository), simply run main.py with the preset hyperparameters. 

Further, control of the architecture and other hyperparameters mainly work through the config file, so changing the architecture (defined as a list of the number of nodes per hidden layer,
so [10, 80, 3] will create 3 hidden layers with 10 nodes in layer 1, 80 nodes in layer 2 and 3 nodes in layer 3) can be done via the config. Input and output layers are defined by the environment.

If you want to train only one time with no variation to hyperparameters or network architecture, set these in the config file and run the LunarLanderMain.py file.

Be aware as results may be overwritten, results in the github repository are the results from the run as described in the report.

CPU vs GPU implementation
Unfortunately, due to the dictionary-based building of the layers in the ActorNetwork and CriticNetwork classes, it is not possible to move the network reliably \
to the GPU, and therefore it is only possible to run the networks on the CPU. An interesting observation is that the CPU implementation seems to be quite a bit \
faster for smaller networks than the GPU implementation.


Credits to MachineLearningwithPhil for programming tutorials
Recommendations:
https://www.youtube.com/watch?v=wc-FxNENg9U&ab_channel=MachineLearningwithPhil
https://www.youtube.com/watch?v=6Yd5WnYls_Y&ab_channel=MachineLearningwithPhil
