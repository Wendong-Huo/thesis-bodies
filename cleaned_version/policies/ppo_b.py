# PPO with body, version 2
# make a one-hot mask for body, don't pass in actual body params. [not completed]

import numpy as np
import gym
from .ppo_nb import PPO_nb

def expand_space(space, env, num):
    if env is None:
        observation = space.sample()
    else:
        observation = env.observation_space.sample()
    low = np.full(observation.shape[0] + num, space.low[0], dtype=np.float32)
    high = np.full(observation.shape[0] + num, space.high[0], dtype=np.float32)
    expaned_space = gym.spaces.Box(low, high, dtype=observation.dtype)
    return expaned_space

class PPO_b(PPO_nb):
    def _setup_model(self) -> None:
        """with body info"""
        self.num_train_bodies = self.env.num_envs
        self.all_params = []
        for i in range(self.num_bodies):
            self.all_params.append(self.env.envs[i].robot.param)

        # expand obs space
        self.observation_space = expand_space(self.observation_space, self.env, self.num_bodies)

        super()._setup_model()