# Wrapper for different bodies with different dimension of obs and action

import gym
import numpy as np

from common import common

class SameDimWrapper(gym.ObservationWrapper):
    max_obs_dim = 0
    max_action_dim = 0
    def __init__(self, env):
        super().__init__(env)
        self.reset_space_size()

    def reset_space_size(self):
        # re-calculate observation space size
        self.old_obs_space = self.observation_space
        self.old_obs_dim = self.old_obs_space.shape[0]
        self.null_dim = self.max_obs_dim-self.old_obs_dim
        assert self.null_dim>=0, f"Max obs dim is not enough. {self.max_obs_dim} < {self.old_obs_dim} "
        low = self.old_obs_space.low[0]
        high = self.old_obs_space.high[0]
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=[self.max_obs_dim])
        # re-calculate action space size
        self.old_action_space = self.action_space
        self.old_action_dim = self.old_action_space.shape[0]
        low = self.old_action_space.low[0]
        high = self.old_action_space.high[0]
        self.action_space = gym.spaces.Box(low=low, high=high, shape=[self.max_action_dim])

    def step(self, action):
        # shorten
        action = action[:self.old_action_dim]
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, obs):
        # expand
        if self.null_dim>0:
            obs = np.concatenate((
                obs,
                [0] * self.null_dim,
            ))
        return obs

def make_same_dim_wrapper(obs_dim, action_dim):
    SameDimWrapper.max_obs_dim = obs_dim
    SameDimWrapper.max_action_dim = action_dim
    return SameDimWrapper