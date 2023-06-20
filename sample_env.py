from replay_memory import MemoryElement
from gym.wrappers import RescaleAction
from utils import torchify

import gym
import numpy as np
import torch

#Taken from https://github.com/gwthomas/Safe-MBPO/ with env document for replicability
def get_env(env_name, render = True):

    if env_name == 'cheetah':
        raise ValueError('Not implemented yet')
    """
    from .env.hopper_no_bonus import HopperNoBonusEnv
    from .env.cheetah_no_flip import CheetahNoFlipEnv
    from .env.ant_no_bonus import AntNoBonusEnv
    from .env.humanoid_no_bonus import HumanoidNoBonusEnv
    envs = {
        'hopper': HopperNoBonusEnv,
        'cheetah-no-flip': CheetahNoFlipEnv,
        'ant': AntNoBonusEnv,
        'humanoid': HumanoidNoBonusEnv
    }
    env = envs[env_name]()
    """
    if render:
        env = gym.make(env_name, render_mode="rgb_array")
    else:
        env = gym.make(env_name)

    if not (np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)):
        env = RescaleAction(env, -1.0, 1.0)
    
    env._max_episode_steps = 1000

    try:
        env.healthy_reward
    except:
        env.healthy_reward = None

    env_name = env_name.lower()

    

    if 'ant' in env_name:
        env.target_entropy = -4
        env.check_done = check_ant_done
    elif 'cheetah' in env_name:
        env.target_entropy = -3
        env.check_done = check_cheetah_done
    elif 'hopper' in env_name:
        env.target_entropy = -1
        env.check_done = check_hopper_done
    elif 'humanoid' in env_name:
        env.target_entropy = -2
        env.check_done = check_humanoid_done
    elif 'pendulum' in env_name:
        env.target_entropy = -1
        env.check_done = check_pendulum_done

    return env

def check_pendulum_done(states):
    states = np.asarray(states)
    if len(states.shape) == 1:
        states = states.reshape(1,-1)
    ang = states[:,1]
    return ~(np.isfinite(states).all(axis=1) & np.abs(ang) <= 0.2)

def check_hopper_done(states):
    
    states = np.asarray(states)
    if len(states.shape) == 1:
        states = states.reshape(1,-1)
    heights, angs = states[:,0], states[:,1]
    return ~(np.isfinite(states).all(axis=1) & (np.abs(states[:,1:]) < 100).all(axis=1) & (heights > .7) & (np.abs(angs) < .2))

def check_cheetah_done(states):
    states = np.asarray(states)
    if len(states.shape) == 1:
        states = states.reshape(1,-1)
    pass

def check_ant_done(states):
    states = np.asarray(states)
    if len(states.shape) == 1:
        states = states.reshape(1,-1)
    heights = states[:,0]
    return ~(np.isfinite(states).all(axis=1) & (heights >= 0.2) & (heights <= 1.0))

def check_humanoid_done(states):
    states = np.asarray(states)
    if len(states.shape) == 1:
        states = states.reshape(1,-1)
    heights = states[:,0]
    return (heights < 1.0) | (heights > 2.0)






class EnvSampler():
    def __init__(self, env_name, env_number = 1 , max_path_length=1000, device = None):


        
        self.env_number = env_number
        self.envs = list(map(get_env, [env_name]*env_number))
        self.path_lengths = [0] * self.env_number
        self.current_states = [None] * self.env_number
        self.dones = [0] * self.env_number

        self.device = None
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.low = torchify(self.action_space.low)
        self.high = torchify(self.action_space.high)
        self.shape = list(self.action_space.shape)

        mpl = self.max_episode()
        if mpl:
            max_path_length = mpl
        self.max_path_length = max_path_length

    def sample(self, agent = None ,eval_t=False):
        for index in range(len(self.envs)):
            env = self.envs[index]
            state = self.current_states[index]
            if state is None:
                self.current_states[index] = env.reset()[0]

        memory_elements = []

        if agent is None:
            action_space = self.envs[0].action_space
            low, high, shape = action_space.low[0], action_space.high[0], action_space.shape[0]

            actions = np.random.uniform(low,high,size = (len(self.envs),shape))

        else:
            actions, log_probs = agent.act(np.asarray(self.current_states), to_cpu = True)

        for index in range(len(self.envs)):
            """
            current_state, env = self.current_states[index], self.envs[index] 

            if agent is None:
                action = env.action_space.sample() 
            else:
                action, log_prob = agent.act(current_state, to_cpu= True)
            """

            action, env = actions[index], self.envs[index]
            next_state, reward, terminal,truncated, info = env.step(action)


            if not env.healthy_reward is None:
                reward = reward - env.healthy_reward 
            #reward = -1*reward
            
            self.path_lengths[index] +=1
            
            current_state = self.current_states[index].copy()

            if terminal or truncated or self.path_lengths[index] >= self.max_path_length:
                self.current_states[index] = None
                self.path_lengths[index] = 0
                truncated = True
            else:
                self.current_states[index] = next_state

            memory_elements.append(MemoryElement(current_state, action, reward, next_state, terminal, truncated))

            
        return memory_elements

    def sample_uniform_action(self, states):
        size = list(states.shape[:-1]) + self.shape
        return self.low + torch.rand(size) * (self.high - self.low)


    def reset(self):
        #self.current_states = list(map(lambda obj: obj.reset(), self.envs))
        self.current_states = [None] * self.env_number
    def partial_reset(indices):
        for i in indices:
            self.current_states[indices] = None #self.envs[i].reset()

    @property
    def env(self):
        return self.envs[0]

    @property
    def dim(self):
        env = self.envs[0]
        return (np.prod(env.observation_space.shape), np.prod(env.action_space.shape))


    def max_episode(self):
        env = self.envs[0]
        if hasattr(env, '_max_episode_steps'):
            return env._max_episode_steps
        elif hasattr(env, 'env'):
            return self.get_max_episode_steps(env.env)
        else:
            return False
    
    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    def check_done(self, states):
        return self.envs[0].check_done(states)
    
    def __len__(self):
        return len(self.envs)
