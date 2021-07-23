import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, lr, input_dims, hidden_dims, output_dims, activation='ReLU'):
        super(Network, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims 
        self.activation = activation

        self.fc1 = nn.Linear(*self.input_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # Set compute device to CUDA GPU if available, use CPU otherwise
        self.to(self.device)
        print("Device: ", self.device)

    def forward(self, state): # Maybe use if statement to enable using different activation functions? (ReLU, tanh, RBF, sigmoid)
        x = F.relu(self.fc1(state))
        actions = F.relu(self.fc2(x))

        return actions

