import gym
import numpy as np

from common import common
from common.seeds import temp_seed

class WalkerWrapper(gym.ObservationWrapper):
    """ unify ant and walker2d, align the observation and action """
    def __init__(self, env: gym.Env):
        if "HopperBulletEnv" in env.spec._env_name:
            self.env_name = "hopper"
            self.num_joint = 3
        elif "HalfCheetahBulletEnv" in env.spec._env_name:
            self.env_name = "halfcheetah"
            self.num_joint = 6
        elif "AntBulletEnv" in env.spec._env_name:
            self.env_name = "ant"
            self.num_joint = 8
        elif "Walker2DBulletEnv" in env.spec._env_name:
            self.env_name = "walker2d"
            self.num_joint = 6
        else:
            raise NotImplementedError
        self.num_feet = len(env.robot.foot_list)

        self.max_num_joint = 8
        self.max_num_feet = 6

        gym.ObservationWrapper.__init__(self, env)
        # re-calculate observation space size
        self.old_obs_space = self.observation_space
        low = self.old_obs_space.low[0]
        high = self.old_obs_space.high[0]
        self.obs_length = 8 + self.max_num_joint*2 + self.max_num_feet
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=[self.obs_length])
        # re-calculate action space size
        self.old_action_space = self.action_space
        low = self.old_action_space.low[0]
        high = self.old_action_space.high[0]
        self.action_space = gym.spaces.Box(low=low, high=high, shape=[self.max_num_joint])


    def step(self, action):
        if self.num_joint < self.max_num_joint:
            action = action[:self.num_joint]
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        """ Align observations! """
        assert observation.shape[0] == 8 + self.num_joint*2 + self.num_feet, "WalkerWrapper: observation dimension error."
        if self.num_joint < self.max_num_joint:
            observation = np.concatenate( (observation[:8+self.num_joint*2] , [0]*(2*(self.max_num_joint-self.num_joint)), observation[8+self.num_joint*2:] ))
        if self.num_feet < self.max_num_feet:
            observation = np.concatenate( (observation, [0]*(self.max_num_feet-self.num_feet) ))
        return observation

class ReAlignedWrapper(gym.ObservationWrapper):
    def __init__(self, env: WalkerWrapper):
        """Before using this wrapper, first wrap with WalkerWrapper"""
        super().__init__(env)
        self.realign_method = common.args.realign_method

        print(self.realign_method)
        if self.realign_method=="general_only":
            self.realign_length = 8
        elif self.realign_method=="joints_only":
            self.realign_length = 16
        elif self.realign_method=="feetcontact_only":
            self.realign_length = 6
        elif self.realign_method=="general_joints":
            self.realign_length = 8+16
        elif self.realign_method=="general_feetcontact":
            self.realign_length = 8+6
        elif self.realign_method=="joints_feetcontact":
            self.realign_length = 16+6
        elif self.realign_method=="general_joints_feetcontact":
            self.realign_length = 8+16+6
        else:
            raise NotImplementedError

        self.realign_idx = np.arange(self.realign_length)
        
        with temp_seed(common.seed):
            np.random.shuffle(self.realign_idx)
        with temp_seed(self.robot.robot_id):
            np.random.shuffle(self.realign_idx)
        if common.args.random_even_same_body:
            with temp_seed(env.rank):
                np.random.shuffle(self.realign_idx)

        if self.realign_method=="general_only":
            self.realign_idx = np.concatenate((
                self.realign_idx,
                np.arange(start=8, stop=8+16+6)
            ))
        elif self.realign_method=="joints_only":
            self.realign_idx = np.concatenate((
                np.arange(start=0,stop=8),
                self.realign_idx + 8,
                np.arange(start=8+16, stop=8+16+6)
            ))
        elif self.realign_method=="feetcontact_only":
            self.realign_idx = np.concatenate((
                np.arange(start=0,stop=8+16),
                self.realign_idx + 8 + 16,
            ))
        elif self.realign_method=="general_joints":
            self.realign_idx = np.concatenate((
                self.realign_idx,
                np.arange(start=8+16,stop=8+16+6),
            ))
        elif self.realign_method=="general_feetcontact":
            self.realign_idx = np.concatenate((
                self.realign_idx[:8],
                np.arange(start=8,stop=8+16),
                self.realign_idx[8:]
            ))
        elif self.realign_method=="joints_feetcontact":
            self.realign_idx = np.concatenate((
                np.arange(start=0,stop=8),
                self.realign_idx
            ))
        elif self.realign_method=="general_joints_feetcontact":
            self.realign_idx = self.realign_idx
        else:
            raise NotImplementedError

        print(self.realign_idx)

    def observation(self, obs):
        # debug:
        # obs = np.arange(30)
        obs = obs[self.realign_idx]
        # print(obs)
        # exit(0)
        return obs
