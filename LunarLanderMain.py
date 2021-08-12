from ddpg import Agent
import gym
import numpy as np
from utils import getConfig, plotLearning, displayTimeEstimate
from datetime import datetime

config_filename = 'config.json'
config = getConfig(config_filename)

env = gym.make(config['settings']['env_name'])

total_episodes = config['settings']['total_episodes']
episodes_interval= config['settings']['episodes_interval']
visual = config['settings']['visual']

agent = Agent(config['settings']['agent'],env)

# changed it to line above where we simply pass the agent object from the config.json
#agent = Agent(alpha=0.00025, beta=0.00025, input_dims=[8], tau=0.001, env=env, \
#    batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

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

print("Done!")
print("Total time: ", datetime.now()-time_checkpoints[0])