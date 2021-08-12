import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime
import json

def plotLearning(scores, filename, x=None, window=5):   
    ''' Plot scores and running average of scores against episodes
    Input
    scores      list    Score values of every episode
    filename    str     Name of the file in which the plot is saved
    x           list    X values against which to plot, if None, list equals length of scores
    window      int     Length of window over which to plot running average
    '''
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg,color='b')
    plt.plot(x, scores,color='r')
    plt.savefig(filename)

def displayTimeEstimate(time_list, episodes_interval=10, total_episodes=1000):
    ''' Keeps and displays the time estimates of the script
    Input
    time_list           list    Array of datetime objects containing the different timesteps at which this function was called
    episodes_interval   int     Number of episodes after which this function is called
    total_episodes      int     Total number of episodes

    Returns
    time_list           list    Array of datetime objects containing the different timesteps at which this function was called, with the new time stamp appended
    '''

    now = datetime.now()
    time_elapsed = now - time_list[-1]
    total_time_elapsed = now - time_list[0]

    print("Time of last", episodes_interval, "episodes:", time_elapsed)
    print("Elapsed time: ", total_time_elapsed)

    episode_number = len(time_list) * episodes_interval
    time_estimate = total_time_elapsed / episode_number * total_episodes
    time_left = time_estimate - total_time_elapsed

    print('Estimated time left: ', time_left)

    time_list.append(now)
    return time_list

def getConfig(filename,verbose=True):
    ''' Get JSON config file with hyperparameters and settings and print in terminal if verbose (for checking)
    Input
    filename    str     Name of the config file including extension
    verbose     bool    If true, prints config contents when called

    Returns
    config      dict    Configuration data
    '''
    with open(filename) as f:
        config = json.load(f)
    
    if verbose:
        print("---CONFIG CONTENTS---")
        print(config)

    return config