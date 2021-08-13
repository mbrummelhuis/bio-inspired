from ddpg import Agent
import gym
import numpy as np
from utils import getConfig, plotLearning, displayTimeEstimate
from datetime import datetime

def LunarLanderMain(config_name):
    config = getConfig(config_name)

    env = gym.make(config['settings']['env_name'])

    total_episodes = config['settings']['total_episodes']
    episodes_interval= config['settings']['episode_interval']
    visual = config['settings']['visual']

    agent = Agent(config['settings']['agent'],env)

    np.random.seed(0)

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
        
        filename = 'lunar-lander.png'
        plotLearning(score_history, filename, window=100)

    agent.save_models() # Save final model parameters

    print("Done training!")
    print("Total time: ", datetime.now()-time_checkpoints[0])