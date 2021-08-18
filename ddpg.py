import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class CriticNetwork(nn.Module):
    def __init__(self, config, input_dims, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.checkpoint_file =self.name+'_ddpg'

        self.number_layers = len(config['hidden_layer_sizes'])
        self.input_dims = input_dims
        self.layers = {}

        for layer in range(self.number_layers+1):
            layer_name = 'fc' + str(layer+1)
            batch_norm_name = 'bn'+ str(layer+1)
            f_val = 'f' + str(layer+1)

            if layer == 0: # If first layer, use input_dims
                self.layers[layer_name] = nn.Linear(self.input_dims, config['hidden_layer_sizes'][0])
                f1 = 1./np.sqrt(self.layers[layer_name].weight.data.size()[0])
                T.nn.init.uniform_(self.layers[layer_name].weight.data, -f1, f1) 
                T.nn.init.uniform_(self.layers[layer_name].bias.data, -f1, f1) 
                self.bn1 = nn.LayerNorm(config['hidden_layer_sizes'][0])
                self.layers[batch_norm_name] = nn.LayerNorm(config['hidden_layer_sizes'][0])

            elif layer == self.number_layers: # If last layer, use n_actions
                self.layers[layer_name] = nn.Linear(n_actions, config['hidden_layer_sizes'][self.number_layers-1])
                f_last = config['f_last']
                self.layers['q'] = nn.Linear(config['hidden_layer_sizes'][self.number_layers-1], 1)
                T.nn.init.uniform_(self.layers[layer_name].weight.data, -f_last, f_last) 
                T.nn.init.uniform_(self.layers[layer_name].bias.data, -f_last, f_last) 
            
            else:
                self.layers[layer_name] = nn.Linear(config['hidden_layer_sizes'][layer-1], \
                    config['hidden_layer_sizes'][layer])
                self.layers[f_val] = 1./np.sqrt(self.layers[layer_name].weight.data.size()[0])
                T.nn.init.uniform_(self.layers[layer_name].weight.data, \
                    -self.layers[f_val], self.layers[f_val]) 
                T.nn.init.uniform_(self.layers[layer_name].bias.data, \
                    -self.layers[f_val], self.layers[f_val]) 
                self.layers[batch_norm_name] = nn.LayerNorm(config['hidden_layer_sizes'][layer])

        self.optimizer = optim.Adam(self.parameters(), lr=config['beta'])

        self.device = 'cpu'
        self.to(self.device)

    def forward(self, state, action):
        state_value = state

        for layer in range(self.number_layers+1):
            layer_name = 'fc' + str(layer+1)
            batch_norm_name = 'bn'+ str(layer+1)  

            if layer < self.number_layers:
                state_value = self.layers[layer_name](state_value)
                state_value = self.layers[batch_norm_name](state_value)
                if layer < self.number_layers-1:
                    state_value = F.relu(state_value)
                else:
                    pass
            
            elif layer == self.number_layers:
                action_value = F.relu(self.layers[layer_name](action))
                state_action_value = F.relu(T.add(state_value, action_value))
                state_action_value = self.layers['q'](state_action_value)
                break

        return state_action_value

    def save_checkpoint(self, save_dir):
        print('... saving checkpoint of network ', self.name)
        filename = os.path.join(save_dir, self.checkpoint_file)
        T.save(self.state_dict(), filename)

    def load_checkpoint(self, save_dir):
        print('... loading checkpoint of network ', self.name)
        filename = os.path.join(save_dir, self.checkpoint_file)
        self.load_state_dict(T.load(filename))

class ActorNetwork(nn.Module):
    def __init__(self, config, input_dims, n_actions, name):
        super(ActorNetwork, self).__init__()

        self.name = name
        self.checkpoint_file = self.name+'_ddpg'

        self.number_layers = len(config['hidden_layer_sizes'])
        self.input_dims = input_dims
        self.layers = {}

        # Here, the network is constructed according to the settings of the config file
        for layer in range(self.number_layers + 1):
            layer_name = 'fc' + str(layer+1)
            batch_norm_name = 'bn'+ str(layer+1)
            f_val = 'f' + str(layer+1)

            if layer == 0: # If first layer, use input_dims
                self.layers[layer_name] = nn.Linear(self.input_dims, config['hidden_layer_sizes'][0])
                f1 = 1./np.sqrt(self.layers[layer_name].weight.data.size()[0])
                T.nn.init.uniform_(self.layers[layer_name].weight.data, -f1, f1) 
                T.nn.init.uniform_(self.layers[layer_name].bias.data, -f1, f1) 
                self.bn1 = nn.LayerNorm(config['hidden_layer_sizes'][0])
                self.layers[batch_norm_name] = nn.LayerNorm(config['hidden_layer_sizes'][0])

            elif layer == self.number_layers: # If last layer, use n_actions
                self.layers[layer_name] = nn.Linear(config['hidden_layer_sizes'][self.number_layers-1], n_actions)
                f_last = config['f_last']
                T.nn.init.uniform_(self.layers[layer_name].weight.data, -f_last, f_last) 
                T.nn.init.uniform_(self.layers[layer_name].bias.data, -f_last, f_last) 
            
            else:
                self.layers[layer_name] = nn.Linear(config['hidden_layer_sizes'][layer-1], \
                    config['hidden_layer_sizes'][layer])
                self.layers[f_val] = 1./np.sqrt(self.layers[layer_name].weight.data.size()[0])
                T.nn.init.uniform_(self.layers[layer_name].weight.data, \
                    -self.layers[f_val], self.layers[f_val]) 
                T.nn.init.uniform_(self.layers[layer_name].bias.data, \
                    -self.layers[f_val], self.layers[f_val]) 
                self.layers[batch_norm_name] = nn.LayerNorm(config['hidden_layer_sizes'][layer])

        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

        self.device = 'cpu'
        self.to(self.device)

    def forward(self, state):
        x = state

        for layer in range(self.number_layers+1):
            layer_name = 'fc' + str(layer+1)
            batch_norm_name = 'bn'+ str(layer+1)
            if layer == self.number_layers:
                break
            else:
                x = self.layers[layer_name](x)
                x = self.layers[batch_norm_name](x)
                x = F.relu(x)

        x = T.tanh(self.layers[layer_name](x))

        return x

    def save_checkpoint(self, save_dir):
        print('... saving checkpoint of network ', self.name)
        filename = os.path.join(save_dir, self.checkpoint_file)
        T.save(self.state_dict(), filename)

    def load_checkpoint(self, save_dir):
        print('... loading checkpoint of network', self.name)
        filename = os.path.join(save_dir, self.checkpoint_file)
        self.load_state_dict(T.load(filename))

class Agent(object):
    def __init__(self,config,env):
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.input_dims = int(env.observation_space.shape[0])
        self.n_actions = int(env.action_space.shape[0])
        self.memory = ReplayBuffer(config['max_mem_size'], self.input_dims, self.n_actions)
        self.batch_size = config['batch_size']

        self.save_dir = config["save_directory"]

        # To keep everything manageable, we only use one network architecture for all networks
        self.actor = ActorNetwork(config['network'], self.input_dims, self.n_actions, name='Actor')
        self.critic = CriticNetwork(config['network'], self.input_dims, self.n_actions, name='Critic')

        self.target_actor = ActorNetwork(config['network'], self.input_dims, self.n_actions, name='TargetActor')
        self.target_critic = CriticNetwork(config['network'], self.input_dims, self.n_actions, name='TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))

        self.update_network_parameters(tau=1)

        print("Compute devices: ", self.actor.device, self.target_actor.device, self.critic.device, self.target_critic.device)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        """
        #Verify that the copy assignment worked correctly
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        critic_state_dict = dict(target_critic_params)
        actor_state_dict = dict(target_actor_params)
        print('\nActor Networks', tau)
        for name, param in self.actor.named_parameters():
            print(name, T.equal(param, actor_state_dict[name]))
        print('\nCritic Networks', tau)
        for name, param in self.critic.named_parameters():
            print(name, T.equal(param, critic_state_dict[name]))
        input()
        """
    def save_models(self):
        self.actor.save_checkpoint(self.save_dir)
        self.target_actor.save_checkpoint(self.save_dir)
        self.critic.save_checkpoint(self.save_dir)
        self.target_critic.save_checkpoint(self.save_dir)

    def load_models(self):
        self.actor.load_checkpoint(self.save_dir)
        self.target_actor.load_checkpoint(self.save_dir)
        self.critic.load_checkpoint(self.save_dir)
        self.target_critic.load_checkpoint(self.save_dir)

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()