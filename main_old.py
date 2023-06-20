import gym
from replay_memory import ReplayMemory, MemoryElement
from sample_env import EnvSampler

from models import SAC, DynamicEnsemble, UniformPolicy



env_name = "HalfCheetah-v4"
seed = 0


replay_size = 1000000


rollout_batch_size = 100
agent_batch_size = 256


num_models = 10




torch.manual_seed(seed)
np.random.seed(seed)


agent = SAC()
dynamic_model = DynamicEnsemble()


#Initialize real environment and real replay buffer
real_env = EnvSampler(env_name)
real_memory = ReplayMemory(replay_size)

#Initialize model buffer
model_memory = ReplayMemory(replay_size)

#Initialize evaluation environment
eval_env = EnvSampler(env_name, env_number = num_models)


E_step = 0
min_reward = np.inf
max_reward = -np.inf


#Initial data collection
#buffer_min = 5000
for _ in range(5000):
    real_memory_element = real_env.sample()[0]
    min_reward = min(real_memory_element.reward, min_reward)
    max_reward = max(real_memory_element.reward, max_reward)

    real_memory.push(real_memory_element)




#Initial model update
#model_initial_steps = 10000
real_memory.fit_states()
for _ in range(10000):
    dynamic_model.update_params(real_memory)
    #DO VIRTUAL AS WELL

#Initial rollout
for _ in range(5000):
    roll_out(dynamic_model, None, real_memory, model_memory)


#steps per epoch * num epoch
for E in range(1000):
    real_memory_element = real_env.sample(agent=agent)[0]
    min_reward = min(real_memory_element.reward, min_reward)
    max_reward = max(real_memory_element.reward, max_reward)

    real_memory.push(real_memory_element)

    if (E+1) %250 == 0:
        real_memory.fit_states()
        for _ in range(2000)
            dynamic_model.update_params(real_memory)
    
    agent.update_C(min_reward, max_reward)

    for n_rollout in range(1):
        roll_out(dynamic_model, agent, real_memory, model_memory)
    
    #solver_updates_per_step = 10
    for n_actor in range(10):
        real_batch_size = agent_batch_size//10
        virtual_batch_size = agent_batch_size - real_batch_size

        memories = real_memory.sample(batch_size = real_batch_size) + model_memory.sample(batch_size = virtual_batch_size)
        agent.update_params(memories)
        

def roll_out(dynamic_model, real_memory, model_memory,agent, real_env , model_batch_size = 100, init = False):
        
    states, actions, rewards, next_states, terminals, truncateds = real_memory.sample_numpy(model_batch_size)

    horizon = agent.horizon
    for t in range(horizon):

        with torch.no_grad():
            if init:
                actions = real_env.sample_uniform_action(states)
                   
                #print(actions.shape)
            else:
                actions, log_prob = agent.act(states, to_cpu = True)
            next_states, rewards = dynamic_model.sample(states, actions, to_cpu = True)

        dones = real_env.check_done(next_states)

        for state, action, reward, next_state, terminal, truncated in zip(states, actions, rewards, next_states, dones,dones):
            mem_el = MemoryElement(state, action, reward, next_state, terminal, truncated)
            model_memory.push(mem_el)

        ixes = np.where(~(dones))[0]
        if len(ixes) == 0:
            break
        states = next_states[ixes]
        
        

def evaluate(agent, eval_env):
    eval_env.reset()

    trajectories = [[] for i in range(len(env))]

    rewards = [0. for i in range(len(env))]
    complete_trajectories = []
    complete_indexes = []
    while True:
        memories = eval_env.sample(agent=agent)

        for ix, mem in enumerate(memories):
            if ix in complete_indexes:
                continue

            trajectories[ix].append(mem)
            rewards[ix] += mem.reward
            rewards[ix]
            if mem.terminal or mem.truncated:
                complete_trajectories.append(trajectories[ix])
                complete_indexes.append(ix)

        if len(complete_indexes) == len(env):
            break

    lengths = np.asarray([len(traj) for traj in complete_trajectories])
    rewards = np.asarray(rewards)

    return rewards.mean(), rewards.std(), lengths.mean(), lengths.std()






"""
if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
    train_predict_model(args, env_memory, predict_env)

    new_rollout_length = set_rollout_length(args, epoch_step)
    if rollout_length != new_rollout_length:
        rollout_length = new_rollout_length
        model_pool = resize_model_pool(args, rollout_length, model_pool)

    rollout_model(args, predict_env, agent, model_pool, env_memory, rollout_length)

cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
env_memory.push(cur_state, action, reward, next_state, done)

if len(env_memory) > args.min_pool_size:
    train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_memory, model_pool, agent)

total_step += 1

if total_step % args.epoch_length == 0:
    '''
    avg_reward_len = min(len(env_sampler.path_rewards), 5)
    avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
    logging.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
    print(total_step, env_sampler.path_rewards[-1], avg_reward)
    '''
    env_sampler.current_state = None
    sum_reward = 0
    done = False
    test_step = 0

    while (not done) and (test_step != args.max_path_length):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
        sum_reward += reward
        test_step += 1
"""