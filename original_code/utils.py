import matplotlib.pyplot as plt 
import numpy as np
import csv

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg,color='b',linewidth=0.8)
    plt.savefig(filename)

def saveScoresAndTime(scores,time,save_dir=None):
    ''' Saves scores to csv file

    Input
    scores      lst     List of scores acquired during training
    time        str     String containing length of time of run
    save_dir    str     Name of the directory in which to save the csv file      
    '''

    if save_dir is not None:
        filename = save_dir
    else:
        filename = 'scores.csv'
    
    with open(filename, 'w') as f:
        write = csv.writer(f)
        write.writerow(scores)
        write.writerow(time)
    
    return