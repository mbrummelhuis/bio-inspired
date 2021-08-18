import os
from ddpg import Agent
import gym
import numpy as np
from utils import getConfig, saveConfig, plotLearning, displayTimeEstimate, saveScoresAndTime
from datetime import datetime
from pathlib import Path

def LunarLanderMain(config_name):
    config = getConfig(config_name)

    Path(config["settings"]["agent"]["save_directory"]).mkdir(parents=True, exist_ok=True)

    saveConfig(config_name)

    env = gym.make(config['settings']['env_name'])

    total_episodes = config['settings']['total_episodes']
    episodes_interval= config['settings']['episode_interval']
    visual = config['settings']['visual']

    agent = Agent(config['settings']['agent'],env)

    np.random.seed(config['settings']['seed'])

    time_checkpoints = [datetime.now()]
    score_history = []

    for episode in range(total_episodes):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            if visual:
                    env.render()
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        
        score_history.append(score)
        print('Episode ', episode, 'Score %.2f' % score, ', 100 game average %.2f' \
            % np.mean(score_history[-100:]))

        if episode % episodes_interval == 0 and episode != 0:
            agent.save_models()
            time_checkpoints = displayTimeEstimate(time_checkpoints, \
                episodes_interval=episodes_interval,total_episodes=total_episodes)
        
        filename = os.path.join(config['settings']['agent']['save_directory'], 'score_plot.png')
        plotLearning(score_history, filename, window=100)

    agent.save_models() # Save final model parameters
    saveScoresAndTime(score_history, str(datetime.now()-time_checkpoints[0]), config['settings']['agent']['save_directory'])

    print("Done training!")
    print("Total time: ", datetime.now()-time_checkpoints[0])

if __name__ == '__main__':
    LunarLanderMain('config.json')