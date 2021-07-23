import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np

class Agent():
    def __init__(self, env, gamma, epsilon, lr, batch_size, input_dims, hidden_dims, output_dims, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.name = 'Continuous agent'
        self.action_space = env.action_space

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = Network(self.lr, input_dims = input_dims, hidden_dims = hidden_dims, output_dims = output_dims)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype = np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)

    def store_transition(self, state, action, reward, state_, done): #state_ is new state
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def getRandomAction(self):
        # Sample a random action from the action space (exploration action)
        action = self.action_space.sample()
        return action
    
    def getAction(self,observation):
        # Get exploitation action
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device) # Store observation in state variable and send to network device (GPU)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action

    def learn(self):
