import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import random

from neural_nets import q_network, p_network, d_network
from utils import torchify,numpyify, get_memories_numpy, get_memories_torch
from replay_memory import MemoryElement


#Based on https://github.com/gwthomas/Safe-MBPO

"""
class Normalizer(Module):
    def __init__(self, dim, epsilon=1e-6):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.zeros(dim))

    def fit(self, X):
        assert torch.is_tensor(X)
        assert X.dim() == 2
        assert X.shape[1] == self.dim
        self.mean.data.copy_(X.mean(dim=0))
        self.std.data.copy_(X.std(dim=0))

    def forward(self, x):
        return (x - self.mean) / (self.std + self.epsilon)

    def unnormalize(self, normal_X):
        return self.mean + (self.std * normal_X)
"""
class DynamicEnsemble():
    def __init__(self, num_ensemble, env_sampler, replay_memory, batch_size = 256, model_steps = 2000, lr = 1e-3,  device = None):
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.min_log_var = -10.0
        self.max_log_var = 1.0
        self.lr = lr

        self.replay_memory = replay_memory

        self.batch_size = batch_size
        self.num_ensemble = num_ensemble

        self.env_sampler = env_sampler
        self.obs_dim, self.act_dim = env_sampler.dim
        self.out_dim = self.obs_dim+1
        self.model_steps = model_steps

        self.optimizer = torch.optim.Adam


        self.max_log_var = nn.Parameter(torch.ones([self.out_dim])*1).to(self.device)
        self.min_log_var = nn.Parameter(torch.ones([self.out_dim])*-10).to(self.device)

        self.ensembles = [d_network(self.obs_dim, self.act_dim, device = self.device) for _ in range(self.num_ensemble)]
        for network in self.ensembles:
            network.optimizer = self.optimizer(network.parameters(), lr = self.lr)


    # Get all models' means on the same set of states and actions
    def means(self, states, actions):
        means, _ = self.forward_all(states, actions)
        return means[:,:,:-1], means[:,:,-1]

    # Get average of models' means
    def mean(self, states, actions):
        next_state_means, reward_means = self.means(states, actions)
        return next_state_means.mean(dim=0), reward_means.mean(dim=0)

    def sample(self, states, actions, to_cpu = False):
        ix = np.random.randint(self.num_ensemble)
        means, log_vars = self.forward_one(states,actions, index = ix)
        stds = torch.exp(log_vars).sqrt()
        samples = means + stds * torch.randn_like(means)

        if to_cpu:
            samples = samples.cpu().detach().numpy()

        return samples[:,:-1], samples[:,-1]

    def forward_one(self, states, actions, index, to_cpu = False):
        states = self.replay_memory.normalize(states)
        if len(states.shape)>1:
            batch_size = states.shape[0]
        else:
            batch_size = 1

        dynamic = self.ensembles[index]

        diffs, log_vars = dynamic.forward(states, actions)

        means = diffs + torch.cat([torchify(states, device = diffs.device), torch.zeros([batch_size, 1], device=diffs.device)], dim=1)

        log_vars = self.max_log_var - F.softplus(self.max_log_var - log_vars)
        log_vars = self.min_log_var + F.softplus(log_vars - self.min_log_var)

        if to_cpu:
            means = means.cpu().detach().numpy()
            log_vars = log_vars.cpu().detach().numpy()

        return means, log_vars
        
        

    def forward_all(self, states, actions, to_cpu = False):
        states = self.replay_memory.normalize(states)
        if len(states.shape)>1:
            batch_size = states.shape[0]
        else:
            batch_size = 1

        means_ = []
        log_vars_ = []
        for dynamic in self.ensembles:
            diffs, log_vars = dynamic.forward(states,actions)

            means = diffs + torch.cat([torchify(states, device = diffs.device), torch.zeros([batch_size, 1], device=diffs.device)], dim=1)
            log_vars = self.max_log_var - F.softplus(self.max_log_var - log_vars)
            log_vars = self.min_log_var + F.softplus(log_vars - self.min_log_var)

            if to_cpu:
                means = means.cpu().detach().numpy()
                log_vars = log_vars.cpu().detach().numpy()


            means_ += [means]
            log_vars_ += [log_vars]

        if to_cpu:
            means_ = np.asarray(means_)
            log_vars_ = np.asarray(log_vars_)
        else:
            means_ = torch.stack(means_, dim= 0)
            log_vars_ = torch.stack(log_vars_, dim=0)
        return means_, log_vars_


    def update_params(self, real_memory):
        
        for ensemble in self.ensembles:
            states, actions, rewards, next_states, terminals, truncateds = real_memory.sample_numpy(self.batch_size)

            targets = torch.cat((torchify(next_states), torchify(rewards)), dim=-1).to(self.device)

            mean, log_std = ensemble.forward(states, actions)
            inv_vars = torch.exp(-log_std).to(self.device)

            squared_error = torch.sum((targets - mean)**2 * inv_vars)
            log_det = torch.sum(log_std).to(self.device)

            loss = torch.mean(squared_error + log_det)

            ensemble.optimizer.zero_grad()
            loss.backward()
            ensemble.optimizer.step()




class SAC():
    def __init__(self,env_sampler, horizon=10, p_learning_rate = 1e-4, q_learning_rate = 3e-4,  device = None):
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #EXPERIMENTAL PARAMETERS
        self.horizon = horizon
        self.gamma = 0.99
        self.tau = 0.005
        self.init_alpha = 1.0
        self.updateC = True
        self.tune_alpha = True
        self.use_log_alpha_loss = True
        self.violation_cost = 0

        self.env_sampler = env_sampler
        self.critic_lr = q_learning_rate
        self.actor_lr = p_learning_rate

        self.optimizer = torch.optim.Adam

        self.log_alpha = torch.tensor(np.log(self.init_alpha), device=device, requires_grad=True)
        self.alpha_optimizer = self.optimizer([self.log_alpha], lr = self.critic_lr)
        self.target_entropy = self.env_sampler.envs[0].target_entropy

        self.obs_dim, self.act_dim = env_sampler.dim
        
        self.critic = q_network(self.obs_dim,self.act_dim, device = self.device)
        self.critic.optimizer = self.optimizer(self.critic.parameters(), lr = self.actor_lr)
        
        self.critic_target = copy.deepcopy(self.critic)
        self.freeze_module(self.critic_target)

        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.actor = p_network(self.obs_dim, self.act_dim, device = self.device)
        self.actor.optimizer = self.optimizer(self.actor.parameters(), lr=self.critic_lr)

        self.criterion = torch.nn.MSELoss()
        torch.autograd.set_detect_anomaly(True)

    def act(self, state, to_cpu = False):
        act, log_prob = self.actor.sample(state)
        if to_cpu:
            return act.cpu().detach().numpy(), log_prob.cpu().detach().numpy()
        else:
            return act, log_prob
    def update_params(self, memories):
        
        states, actions, rewards, next_states, terminals, truncateds = get_memories_torch(memories, device = self.device)

        next_q = self.get_critic_target_prediction(next_states, rewards, terminals)
        q1, q2 = self.critic(states,actions)
        q_loss = self.get_critic_loss(q1,q2,next_q)
        p_loss, log_pi = self.get_actor_loss(states)        
        alpha_loss = self.get_alpha_loss(log_pi)

        self.backpropagate(q_loss, p_loss, alpha_loss)
        self.soft_update()

    def freeze_module(self,module):
        for p in module.parameters():
            p.requires_grad = False

    def update_C(self, r_min, r_max):
        self.r_min, self.r_max = r_min, r_max
        if self.updateC:
            self.violation_cost = (self.r_max - self.r_min) / self.gamma**self.horizon - self.r_max

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def violation_value(self):
        return -self.violation_cost / (1. - self.gamma)

    
    def get_critic_target_prediction(self, next_states, rewards, terminals):
        with torch.no_grad():
            next_actions, next_log_pis = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            min_qn = torch.min(q1_next,q2_next) - self.alpha.detach()*next_log_pis
            next_q = rewards + self.gamma*(~terminals*min_qn)
            done_indexes = np.where(terminals.cpu())
            next_q[done_indexes] = self.violation_value # rewards + self.gamma * self.violation_value

            return next_q
    
    def get_critic_loss(self, q1,q2,next_q):
        q_loss_1 = self.criterion(q1, next_q)
        q_loss_2 = self.criterion(q2, next_q)
        q_loss = q_loss_1+q_loss_2

        return q_loss

    def get_actor_loss(self, states):
        p_action, log_pi = self.actor.sample(states)
        qpi1, qpi2 = self.critic(states, p_action)

        #Randomy choose
        min_qp = random.choice( (qpi1,qpi2))
        #min_qp = torch.min(qpi1, qpi2)
        p_loss = (self.alpha * log_pi - min_qp).sum()

        return p_loss, log_pi

    def get_alpha_loss(self, log_pi):
        
        alpha_loss = None
        if self.tune_alpha:
            multiplier = self.log_alpha if self.use_log_alpha_loss else self.alpha
            alpha_loss = -multiplier * torch.mean(log_pi.detach() + self.target_entropy)
        return alpha_loss

    def backpropagate(self, q_loss, p_loss, alpha_loss):
        optimizers = [ self.actor.optimizer, self.critic.optimizer,self.alpha_optimizer]
        losses = [ p_loss,q_loss, alpha_loss]

        if alpha_loss is None:
            optimizer = optimizer[:-1]
            losses = losses[:-1]

        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def soft_update(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


class UniformPolicy():
    def __init__(self, env_sampler, device = None):
    
        self.device = None
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        action_space = env_sampler.action_space
        self.low = torchify(action_space.low, device=self.device)
        self.high = torchify(action_space.high, device=self.device)
        self.shape = list(action_space.shape)


    def act(self, states, eval):
        batch_size = len(states)
        
        return self.low + torch.rand(batch_size, *self.shape, device=self.device) * (self.high - self.low)

    def prob(self, actions):
        batch_size = len(actions)
        
            
        p = 1./torch.prod(self.high - self.low)
        return torch.full([batch_size], p, device=self.device)

    def log_prob(self, actions):
        return torch.log(self.prob(actions))