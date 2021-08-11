from ddpg_phil import Agent
import gym
import numpy as np
from utils import plotLearning
from datetime import datetime

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.00025, beta=0.00025, input_dims=[8], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

begin_time = datetime.now()
time_checkpoints = []
time_checkpoints.append(begin_time)

score_history = []
for i in range(1000):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
    
    score_history.append(score)
    print('Episode ', i, 'Score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))

    if i % 25 == 0:
        agent.save_models()
    
    filename = 'lunar-lander.png'
    plotLearning(score_history, filename, window=100)