import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
from network import Network

class Agent():
    def __init__(self, gamma, epsilon, lr, batch_size, input_dims, hidden_dims, output_dims, max_mem_size=100000, eps_end=0.01, eps_dec=5e-5):
        self.action_space = [i for i in range(output_dims)]

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
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)

    def storeTransition(self, state, action, reward, state_, done): #state_ is new state
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

        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad() # Set gradient of optimizer to zero

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype = np.int32)

        # Bring batches of all variables to GPU
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch) # Use target network here when you want to improve
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min
