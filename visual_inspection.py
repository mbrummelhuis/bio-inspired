from utils import getConfig
from ddpg import Agent
import gym
import json
import time
import os

"""
Script for doing a visual inspection of 
"""

filename = 'config.json'

with open(filename) as f:
    config = json.load(f)

config["settings"]["agent"]["network"]["hidden_layer_sizes"] = [400, 300] # Set appropriate hidden layer sizes (must correspond to loaded network)
config["settings"]["visual"] = True
config["settings"]["agent"]["save_directory"] = os.path.join('results_main2', 'Results_main_a5') # Put path to load files here
json_file = open(filename, "w")
json.dump(config, json_file)
json_file.close()
time.sleep(5)

env = gym.make('LunarLanderContinuous-v2')
config = getConfig(filename)
agent = Agent(config["settings"]["agent"],env)
agent.load_models()

visual = config['settings']['visual']

for i in range(30):
    obs = env.reset()
    done = False
    while not done:
        if visual:
                env.render()
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        obs = new_state
