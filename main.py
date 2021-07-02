import gym
from agent import Agent

env_name = "LunarLanderContinuous-v2" # Set environment
number_episodes = 200 # Sets number of episodes

# Create environment specified earlier and reset
env = gym.make(env_name)
env.reset()

# Set variables to initial state
done = False
total_reward = 0

# Initialise agent
agent = Agent(env)

# Main running loops, outer loop is episodes and inner loop is frames in each episode
for episode in range(number_episodes):
    while done == False:
        action = agent.getRandomAction(env)
        print('Action: ', action)
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
        total_reward += reward

print("Total reward: ", total_reward)
env.close()

