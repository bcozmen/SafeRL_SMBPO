import math
import torch
import numpy as np



def get_memories_numpy(memories):
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

def get_memories_torch(memories, device = None):
    this = get_memories_numpy(memories)
    l = []
    for t in this:
        torched = torchify(t, device = device)
        if len(torched.shape)==1:
            torched = torched[:,None]
        l.append(torched)
    return tuple(l)


def torchify(x, double = True,  device=None):
    if torch.is_tensor(x):
        pass

    elif isinstance(x, np.ndarray):
        try:
            x = torch.from_numpy(x)
        except:
            print(x)
            for i in x:
                if isinstance(i, tuple):
                    print(i)
                    le()
            le()
    else:
        x = torch.tensor(x)

    if x.dtype == torch.double or x.dtype == torch.float:
        if double:
            x = x.double()
        else:
            x = x.float()

    
    if not device is None:
        x = x.to(device)

    return x

def numpyify(x):
    if isinstance(x, np.ndarray):
        return x
    
    elif torch.is_tensor(x):
        return x.cpu().numpy()
    
    else:
        return np.array(x)