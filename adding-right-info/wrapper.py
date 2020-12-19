import gym
import numpy as np


class BodyinfoWrapper(gym.ObservationWrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.
    :param env: the environment
    :param width:
    :param height:
    """

    def __init__(self, env: gym.Env, bodyinfo=0):
        gym.ObservationWrapper.__init__(self, env)
        self.old_obs_space = self.observation_space
        low = np.append(self.old_obs_space.low, self.old_obs_space.low[0])
        high = np.append(self.old_obs_space.high, self.old_obs_space.high[0])
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.bodyinfo = bodyinfo

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.append(obs, self.bodyinfo)
        return obs
