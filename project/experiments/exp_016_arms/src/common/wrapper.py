import gym
import numpy as np

from common import common
from common.seeds import temp_seed

class ReAlignedWrapper(gym.ObservationWrapper):
    """Realign without any zero-padding"""

    def __init__(self, env):
        # assert not isinstance(env, WalkerWrapper), "Don't padding zero before realign"
        super().__init__(env)
        self.realign_method = common.args.realign_method
        self.num_feet = len(env.robot.foot_list)
        self.num_joints = env.action_space.shape[0]
        self.length_general = 8
        self.length_joints = self.num_joints * 2
        self.length_feet = self.num_feet
        # print(self.num_feet, self.num_joints)
        print(self.realign_method)
        if self.realign_method == "aligned":
            self.realign_length = 0
        elif self.realign_method == "general_only":
            self.realign_length = self.length_general
        elif self.realign_method == "joints_only":
            self.realign_length = self.length_joints
        elif self.realign_method == "feetcontact_only":
            self.realign_length = self.length_feet
        elif self.realign_method == "general_joints":
            self.realign_length = self.length_general+self.length_joints
        elif self.realign_method == "general_feetcontact":
            self.realign_length = self.length_general+self.length_feet
        elif self.realign_method == "joints_feetcontact":
            self.realign_length = self.length_joints+self.length_feet
        elif self.realign_method == "general_joints_feetcontact":
            self.realign_length = self.length_general+self.length_joints+self.length_feet
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
        if self.realign_method == "aligned":
            self.realign_idx = np.arange(start=0,stop=self.length_general+self.length_joints+self.length_feet)
        elif self.realign_method == "general_only":
            self.realign_idx = np.concatenate((
                self.realign_idx,
                np.arange(start=self.length_general, stop=self.length_general+self.length_joints+self.length_feet)
            ))
        elif self.realign_method == "joints_only":
            self.realign_idx = np.concatenate((
                np.arange(start=0, stop=self.length_general),
                self.realign_idx + self.length_general,
                np.arange(start=self.length_general+self.length_joints, stop=self.length_general+self.length_joints+self.length_feet)
            ))
        elif self.realign_method == "feetcontact_only":
            self.realign_idx = np.concatenate((
                np.arange(start=0, stop=self.length_general+self.length_joints),
                self.realign_idx + self.length_general + self.length_joints,
            ))
        elif self.realign_method == "general_joints":
            self.realign_idx = np.concatenate((
                self.realign_idx,
                np.arange(start=self.length_general+self.length_joints, stop=self.length_general+self.length_joints+self.length_feet),
            ))
        elif self.realign_method == "general_feetcontact":
            for i, v in enumerate(self.realign_idx):
                if v>=self.length_general:
                    self.realign_idx[i] = v+self.length_joints
            self.realign_idx = np.concatenate((
                self.realign_idx[:self.length_general],
                np.arange(start=self.length_general, stop=self.length_general+self.length_joints),
                self.realign_idx[self.length_general:]
            ))
        elif self.realign_method == "joints_feetcontact":
            self.realign_idx = np.concatenate((
                np.arange(start=0, stop=self.length_general),
                self.realign_idx + self.length_general
            ))
        elif self.realign_method == "general_joints_feetcontact":
            self.realign_idx = self.realign_idx
        else:
            raise NotImplementedError
        print(self.realign_idx)

    def observation(self, obs):
        return obs[self.realign_idx]
