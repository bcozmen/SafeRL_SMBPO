from utils import torchify,numpyify

import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.distributions import Normal

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


#Dynamics network - encoding the environment dynamics
class d_network(nn.Module):
    def __init__(self, num_state, num_actions, hidden_dim = 200, device = None):
        super(d_network, self).__init__()
        
        self.num_state = num_state
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.network = nn.Sequential(
                nn.Linear(num_state + num_actions, hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, (num_state+1)*2)
            )
        self.double()
        self.apply(weights_init_)
        self.to(device)

    def forward(self,state,action):
        state = torchify(state, device = self.device)
        action = torchify(action, device = self.device)
        x = torch.cat([state, action], dim=-1)
        
        x = self.network(x)
        mean,log_std = x[...,:self.num_state+1],x[...,self.num_state+1:]

        return mean,log_std

    def sample(self, state, action):
        mean, std_log = self.forward(state,action)
        std = std_log.exp()

        return tdist.Normal(mean,std).sample()

class q_network(nn.Module):
    def __init__(self, num_state, num_actions, hidden_dim = 256, device = None):
        super(q_network, self).__init__()

        self.num_state = num_state
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.network1 = nn.Sequential(
                nn.Linear(num_state + num_actions, hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, 1)
        )
        
        self.network2 = nn.Sequential(
                nn.Linear(num_state + num_actions, hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, 1)
        )
        
        self.double()
        self.apply(weights_init_)
        self.to(device)

    def forward(self,state,action):
        state = torchify(state, device = self.device)
        action = torchify(action, device = self.device)
        
        x = torch.cat([state, action], dim=-1)
        
        r1,r2 = self.network1(x), self.network2(x)
        return r1,r2
    
class p_network(nn.Module):
    #policy 0 = gaussian
    #policy 1 = deterministic
    def __init__(self, num_state, num_actions, hidden_dim = 256, device = None):
        super(p_network, self).__init__()

        self.num_state = num_state
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.network = nn.Sequential(
            nn.Linear(num_state, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, num_actions*2)
        )
                
        self.apply(weights_init_)
        self.double()
        self.to(device)
            
    def forward(self,state):
        state = torchify(state, device = self.device)
            
        x = self.network(state)
        mean,log_std = x[...,:self.num_actions],x[...,self.num_actions:]
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean,std)
        action = normal.sample()

        log_prob = normal.log_prob(action)
        log_prob = torch.sum(log_prob, dim=-1)

        if len(log_prob.shape) == 1:
            log_prob = log_prob[:,None]

        action = torch.tanh(action)

        
        return action, log_prob