import gym
import agent

env_name = "LunarLanderContinuous-v2"
env = gym.make(env_name)
env.reset()
print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)
done = False
total_reward = 0
while done == False:
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    print(done)
    print(reward)
    total_reward += reward
print("Total reward: ", total_reward)
env.close()

