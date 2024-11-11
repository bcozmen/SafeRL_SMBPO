import random
import numpy as np

from utils import torchify

class MemoryElement:
    def __init__(self, state, action, reward, next_state, terminal, truncated):
        self.state = np.asarray(state)
        self.action = np.asarray(action)
        self.reward = reward
        self.next_state = np.asarray(next_state)
        self.terminal = terminal
        self.truncated = truncated



    



class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.state_buffer = []

        self.epsilon = 1e-6

    #Push a memory element or a list of memory elements into the buffer
    def push(self, memory_element):
        if isinstance(memory_element, MemoryElement):
            self.buffer.append(memory_element)
            self.state_buffer.append(memory_element.state)
            

        elif isinstance(memory_element, list):
            if all(isinstance(x, MemoryElement) for x in memory_element):
                self.buffer = self.buffer + memory_element
                for mem in memory_element:
                    self.state_buffer.append(mem.state)
            
            else:
                raise ValueError('All elements fed into ReplayMemory should be an instance of MemoryElement - code 1')
        else:
            raise ValueError('All elements fed into ReplayMemory should be an instance of MemoryElement - code 2')
        value = len(self.buffer) - self.capacity
        if value > 0:
            self.buffer = self.buffer[value:]
            self.state_buffer = self.state_buffer[value:]
    
    def fit_states(self):
        self.state_mean = np.mean(self.state_buffer, axis=0)
        self.state_std = np.std(self.state_buffer, axis=0)

    def normalize(self, x):
        return (x - self.state_mean) / (self.state_std + self.epsilon)

    def unnormalize(self, normal_X):
        return self.state_mean + (self.state_std * normal_X)
    #Sample randomly from replay memory
    def sample(self, batch_size = 256):
        if batch_size> len(self.buffer):
            batch_size = len(self.buffer)
        return random.sample(self.buffer, batch_size)

    def sample_numpy(self, batch_size = 256):
        mems = self.sample(batch_size=batch_size)
        return self.get_memories_numpy(mems)

    def sample_torch(self, batch_size=256, device = None):
        mems = self.sample(batch_size=batch_size)
        return self.get_memories_torch(mems, device=device)
    
    def get_memories_numpy(self, memories):
        states, actions, rewards, next_states, terminals, truncateds = [],[],[],[],[],[]
        for mem in memories:
            states.append(mem.state)
            actions.append(mem.action)
            rewards.append(mem.reward)
            next_states.append(mem.next_state)
            terminals.append(mem.terminal)
            truncateds.append(mem.truncated)

        states, actions, rewards = np.asarray(states),np.asarray(actions),np.asarray(rewards)[:, None]
        next_states, terminals, truncateds = np.asarray(next_states),np.asarray(terminals)[:,None],np.asarray(truncateds)[:, None]

        return states, actions, rewards, next_states, terminals, truncateds

    def get_memories_torch(self, memories, device = None):
        this = get_memories_numpy(memories)
        l = []
        for t in this:
            torched = torchify(t, device = device)
            if len(torched.shape)==1:
                torched = torched[:,None]
            l.append(torched)
        return tuple(l)

    def __len__(self):
        return len(self.buffer)

