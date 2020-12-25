import gym
import numpy as np


class BodyinfoWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, bodyinfo=0):
        # one-hot encoding length for bodyinfo: 10
        self.bodyinfo_length = 2

        gym.ObservationWrapper.__init__(self, env)
        self.old_obs_space = self.observation_space
        low = self.old_obs_space.low[0]
        high = self.old_obs_space.high[0]
        obs_length = len(self.old_obs_space.high) + self.bodyinfo_length
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=[obs_length])

        self.bodyinfo = np.zeros(shape=[self.bodyinfo_length], dtype=np.float32)
        self.bodyinfo[bodyinfo] = 1

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.concatenate((obs, self.bodyinfo))
        return obs
