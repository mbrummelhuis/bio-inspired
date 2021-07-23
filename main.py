import gym
from agent import Agent
import numpy as np

def main(verbose = True):
    env_name = "LunarLander-v2" # Set environment
    number_episodes = 250 # Sets number of episodes
    scores, eps_history = [],[]
    # Create environment specified earlier and reset
    env = gym.make(env_name)
    env.reset()

    # Initialise agent
    agent = Agent(0.99, 1.0, 0.005, 64, [8], 64, 4)

    # Main running loops, outer loop is episodes and inner loop is frames in each episode
    for episode in range(number_episodes):
        # Set variables to initial episode state
        done = False
        ep_reward = 0
        observation = env.reset()

        while done == False:
            action = agent.getAction(observation)
            observation_, reward, done, info = env.step(action)
            ep_reward += reward
            agent.storeTransition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(ep_reward)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if verbose:
            print("Episode ", episode, "Episode reward: ", ep_reward, "Epsilon ", agent.epsilon)

    env.close()
    return

if __name__ == '__main__':
    main(True) # Set true for verbose

