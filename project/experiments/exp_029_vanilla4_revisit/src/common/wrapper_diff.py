# Wrappers for different topologies (observation has different dimensionalities)
import sys
import gym
import numpy as np

from common import common
from common import gym_interface
def get_wrapper_class():
    args = common.args
    body_set = set()
    if len(args.train_bodies)>0:
        for body in args.train_bodies:
            body_set.add(gym_interface.template(body).capitalize())
    else:
        for body in args.test_bodies:
            body_set.add(gym_interface.template(body).capitalize())

    assert len(body_set)==2, "Need two types of bodies to apply this wrapper"
    classname = ''.join(sorted(body_set))
    print(classname)

    classname = f"{classname}Case{common.args.wrapper_case}" # e.g. Walker2DWalkerArmsCase1
    return getattr(sys.modules[__name__], classname)

# walkerarms: 8+8*2+2=26
# walker2d: 8+6*2+2=22
# hopper: 8+3*2+1=15

class TwoTopologiesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.observation_space.shape[0]==self.setting["length"][0] or self.observation_space.shape[0]==self.setting["length"][1]
        self.num_feet = len(env.robot.foot_list)
        self.num_joints = env.action_space.shape[0]
        self.max_num_joints = max(self.setting["joints"])
        self.max_num_feet = max(self.setting["feet"])
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
        
        self.debug = 1


    def step(self, action):
        if self.debug:
            print(self.action_realign_idx)
            print(self.obs_realign_idx)
            self.debug=0
        if self.num_joints < self.max_num_joints:
            action = action[self.action_realign_idx]
            action = action[:self.num_joints]
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, obs):
        if obs.shape[0]==max(self.setting["length"]): # Walker2D
            return obs
        else: # Hopper
            obs = np.concatenate((
                obs,
                [0]*(self.setting["length"][0]-self.setting["length"][1]), # padding zero
            ))
            obs = obs[self.obs_realign_idx] # realign observation into different cases
            assert obs.shape[0]==max(self.setting["length"])
            return obs
    

class Walker2dWalkerarmsWrapper(TwoTopologiesWrapper):
    def __init__(self, env):
        self.setting = {
            "length": [26,22],
            "joints": [8,6],
            "feet":   [2,2],
        }
        super().__init__(env)

class Walker2dWalkerarmsCase1(Walker2dWalkerarmsWrapper):
    def __init__(self, env):
        super().__init__(env)
        #Case1(Walker2D)
        # obs = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19|20,21,--,--,--,--]
        # act = [0,1,2,3,4,5,-,-]
        self.obs_realign_idx = [0,1,2,3,4,5,6,7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
        self.action_realign_idx = [0,1,2,3,4,5,6,7]
class Walker2dWalkerarmsCase2(Walker2dWalkerarmsWrapper):
    def __init__(self, env):
        super().__init__(env)
        #Case2(Walker2D) 
        # obs = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19,--,--,--,--|20,21]
        # act = [-,-,0,1,2,3,4,5]
        self.obs_realign_idx = [0,1,2,3,4,5,6,7, 8, 9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,20,21]
        self.action_realign_idx = [2,3,4,5,6,7,0,1]

class Walker2dWalkerarmsCase3(Walker2dWalkerarmsWrapper):
    def __init__(self, env):
        super().__init__(env)
        #Case3(Walker2D) 
        # obs = [0,1,2,3,4,5,6,7|--,--,--,--,08,09,10,11,12,13,14,15,16,17,18,19|20,21]
        # act = [-,-,0,1,2,3,4,5]
        self.obs_realign_idx = [0,1,2,3,4,5,6,7,22,23,24,25, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21]
        self.action_realign_idx = [2,3,4,5,6,7,0,1]


class Walker2DHopperWrapper(TwoTopologiesWrapper):
    def __init__(self, env):
        self.setting = {
            "length": [22,15],
            "joints": [6,3],
            "feet":   [2,1],
        }
        super().__init__(env)
    

class Walker2DHopperCase1(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 1: obs(Hopper)   = [0,1,2,3,4,5,6,7|10,11,12,13,08,09|14|--,--,--,--,--,--,--]
        self.obs_realign_idx = [0,1,2,3,4,5,6,7,10,11,12,13, 8, 9,14,15,16,17,18,19,20,21]

class Walker2DHopperCase2(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 2: obs(Hopper)   = [0,1,2,3,4,5,6,7|12,13,08,09,10,11|14|--,--,--,--,--,--,--]
        self.obs_realign_idx = [0,1,2,3,4,5,6,7,12,13, 8, 9,10,11,14,15,16,17,18,19,20,21]

class Walker2DHopperCase3(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 3: obs(Hopper)   = [0,1,2,3,4,5,6,7|10,11,12,13,08,09,--,--,--,--,--,--|14,--]
        self.obs_realign_idx = [0,1,2,3,4,5,6,7,10,11,12,13, 8, 9,15,16,17,18,19,20,14,21]

class Walker2DHopperCase4(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 4: obs(Hopper)   = [0,1,2,3,4,5,6,7|12,13,08,09,10,11,--,--,--,--,--,--|14,--]
        self.obs_realign_idx = [0,1,2,3,4,5,6,7,12,13, 8, 9,10,11,15,16,17,18,19,20,14,21]


class Walker2DHopperCase5(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 5: obs(Hopper)   = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,--,--,--,--,--,--|14,--]
        self.obs_realign_idx = [0,1,2,3,4,5,6,7, 8, 9,10,11,12,13,15,16,17,18,19,20,14,21]

class Walker2DHopperCase6(Walker2DHopperWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Case 6: obs(Hopper)   = [0,1,2,3,4,5,6,7|--,--,--,--,--,--,08,09,10,11,12,13|14,--]
        self.obs_realign_idx = [0,1,2,3,4,5,6,7,15,16,17,18,19,20, 8, 9,10,11,12,13,14,21]
