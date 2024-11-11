import gym



class Env(gym.Wrapper):
    def __init__(self, env_name):

        super().__init__(gym.make(env_name))
        self._max_episode_steps = 1000
        if not (np.all(self.env.action_space.low == -1.0) and np.all(self.env.action_space.high == 1.0)):
            self.env = RescaleAction(self.env, -1.0, 1.0)


class HopperEnv(Env):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.target_entropy = -1
        self._max_episode_steps = 1000
    
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.env.healthy_reward

        return obs, reward, terminated, truncated, info

class HopperEnv(GymHopperEnv, MujocoWrapper):
    def __init__(self):

        

    @staticmethod
    def check_done(states):
        heights, angs = states[:,0], states[:,1]
        return ~(np.isfinite(states).all(axis=1) & (np.abs(states[:,1:]) < 100).all(axis=1) & (heights > .7) & (np.abs(angs) < .2))

    def qposvel_from_obs(self, obs):
        qpos = self.sim.data.qpos.copy()
        qpos[1:] = obs[:5]
        qvel = obs[5:]
        return qpos, qvel


class HopperNoBonusEnv(HopperEnv):


    def step(self, action):
        next_state, reward, done, info = super().step(action)
        reward -= 1     # subtract out alive bonus
        info['violation'] = done
        return next_state, reward, done, info

    def check_done(self, states):
        return self.check_violation(states)

    def check_violation(self, states):
        heights, angs = states[:,0], states[:,1]
        return ~(np.isfinite(states).all(axis=1) & (np.abs(states[:,1:]) < 100).all(axis=1) & (heights > .7) & (np.abs(angs) < .2))