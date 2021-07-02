import gym
from agent import Agent

env_name = "LunarLanderContinuous-v2" # Set environment
number_episodes = 10 # Sets number of episodes

# Create environment specified earlier and reset
env = gym.make(env_name)
env.reset()

# Initialise agent
agent = Agent(env)

# Main running loops, outer loop is episodes and inner loop is frames in each episode
for episode in range(number_episodes):
    # Set variables to initial episode state
    print("Episode ", episode)
    env.reset()
    done = False
    total_reward = 0
    while done == False:
        action = agent.getRandomAction()
        state, reward, done, info = env.step(action)
        env.render()
        total_reward += reward

    print("Total reward: ", total_reward)

env.close()

