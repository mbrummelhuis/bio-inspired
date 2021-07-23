import torch as T
import torch.nn as nn
import torch.optim as optim

class Network(nn.module):
    def __init__(self, lr, input_dims, hidden_dims, output_dims, activation):
        super(Network, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.activation = activation

        self.fc1 = nn.Linear(self.input_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.output_dims)
        self.optimizer = optim.Adam(self.parameters, lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # Set compute device to CUDA GPU if available, use CPU otherwise
        self.to(self.device)

    def forward(self, state): # Maybe use if statement to enable using different activation functions? (ReLU, tanh, RBF, sigmoid)
        x = self.


class Agent():
    def __init__(self, env, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
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

        self.Q_eval = 


    def getRandomAction(self):
        # Sample a random action from the action space
        action = self.action_space.sample()
        return action
    
    def getAction(self,state,policy):
        # Sample an action based on the policy and state
        
        return action