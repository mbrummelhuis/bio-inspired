import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime

def plotLearning(scores, filename, x=None, window=5):   
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
    now = datetime.now()
    time_elapsed = now - time_list[-1]
    total_time_elapsed = now - time_list[0]

    print("Time of last ", episodes_interval, " episodes: ", time_elapsed)
    print("Elapsed time: ", total_time_elapsed)

    episode_number = len(time_list) * episodes_interval
    time_estimate = total_time_elapsed / episode_number * total_episodes
    time_left = time_estimate - total_time_elapsed

    print('Estimated time left: ', time_left)

    time_list.append(now)
    return time_list