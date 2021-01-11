# Wrappers for different topologies (observation has different dimensionalities)
import gym
import numpy as np

from common import common

# walker2d: 8+6*2+2=22
# hopper: 8+3*2+1=15
class Walker2DHopperWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.observation_space.shape[0]==22 or self.observation_space.shape[0]==15
        self.num_feet = len(env.robot.foot_list)
        self.num_joints = env.action_space.shape[0]
        self.max_num_joints = max(6,3)
        self.max_num_feet = max(2,1)
        # re-calculate observation space size
        self.old_obs_space = self.observation_space
        low = self.old_obs_space.low[0]
        high = self.old_obs_space.high[0]
        self.obs_length = 8 + self.max_num_joints*2 + self.max_num_feet
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=[self.obs_length])
        # re-calculate action space size
        self.old_action_space = self.action_space
        low = self.old_action_space.low[0]
        high = self.old_action_space.high[0]
        self.action_space = gym.spaces.Box(low=low, high=high, shape=[self.max_num_joints])
    
    def step(self, action):
        if self.num_joints < self.max_num_joints:
            action = action[:self.num_joints]
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, obs):
        if obs.shape[0]==22: # Walker2D
            return obs
        else: # Hopper
            obs = np.concatenate((
                obs,
                [0]*(22-15), # padding zero
            ))
            obs = obs[self.hopper_realign_idx] # realign observation into different cases
            assert obs.shape[0]==22
            return obs

class Walker2DHopperCase1(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 1: obs(Hopper)   = [0,1,2,3,4,5,6,7|10,11,12,13,08,09|14|--,--,--,--,--,--,--]
        self.hopper_realign_idx = [0,1,2,3,4,5,6,7,10,11,12,13, 8, 9,14,15,16,17,18,19,20,21]

class Walker2DHopperCase2(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 2: obs(Hopper)   = [0,1,2,3,4,5,6,7|12,13,08,09,10,11|14|--,--,--,--,--,--,--]
        self.hopper_realign_idx = [0,1,2,3,4,5,6,7,12,13, 8, 9,10,11,14,15,16,17,18,19,20,21]

class Walker2DHopperCase3(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 3: obs(Hopper)   = [0,1,2,3,4,5,6,7|10,11,12,13,08,09,--,--,--,--,--,--|14,--]
        self.hopper_realign_idx = [0,1,2,3,4,5,6,7,10,11,12,13, 8, 9,15,16,17,18,19,20,14,21]

class Walker2DHopperCase4(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 4: obs(Hopper)   = [0,1,2,3,4,5,6,7|12,13,08,09,10,11,--,--,--,--,--,--|14,--]
        self.hopper_realign_idx = [0,1,2,3,4,5,6,7,12,13, 8, 9,10,11,15,16,17,18,19,20,14,21]


class Walker2DHopperCase5(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 5: obs(Hopper)   = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,--,--,--,--,--,--|14,--]
        self.hopper_realign_idx = [0,1,2,3,4,5,6,7, 8, 9,10,11,12,13,15,16,17,18,19,20,14,21]

class Walker2DHopperCase6(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 6: obs(Hopper)   = [0,1,2,3,4,5,6,7|--,--,--,--,--,--,08,09,10,11,12,13|14,--]
        self.hopper_realign_idx = [0,1,2,3,4,5,6,7,15,16,17,18,19,20, 8, 9,10,11,12,13,14,21]
