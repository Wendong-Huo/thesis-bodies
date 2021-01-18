# Wrappers for 900 mutants, different topology, same dimension

import sys
import gym
import numpy as np

from common import common
from common import gym_interface

class MutantWrapper(gym.ObservationWrapper):
    """Mutants 900, different topology, but same dimension for simplicity."""
    def __init__(self, env):
        super().__init__(env)
        self.debug = 1
        assert gym_interface.template(env.robot.robot_id)=="walkerarms", "MutantWrapper only support walkerarms for now."
        self.default_joint_order = {"arm_joint":0, "arm_left_joint":1, "thigh_joint":2, "leg_joint":3, "foot_joint":4, "thigh_left_joint":5, "leg_left_joint":6, "foot_left_joint":7}
        self.default_foot_order = {"foot":0, "foot_left":1}
        self.action_realign_idx = []
        self.obs_realign_idx = [0,1,2,3,4,5,6,7]
        env.reset()
        print(f"{env.robot.robot_id} > init > joints:")
        for i,j in enumerate(self.robot.ordered_joints):
            print(f"\t{j.joint_name}")
            self.obs_realign_idx.append(8+self.default_joint_order[j.joint_name]*2)
            self.obs_realign_idx.append(8+self.default_joint_order[j.joint_name]*2+1)
            self.action_realign_idx.append(self.default_joint_order[j.joint_name])
        print(f"{env.robot.robot_id} > init > feet:")
        for f in self.robot.foot_list:
            print(f"\t{f}")
            self.obs_realign_idx.append(24+self.default_foot_order[f])
        # obs ------------ F() ------> abs_obs
        # abs_action --- F^{-1}() ---> action
        self.obs_realign_idx = np.argsort(self.obs_realign_idx).tolist()

    def step(self, action):
        if self.debug:
            print(self.action_realign_idx)
            print(self.obs_realign_idx)
            self.debug=0
        action = action[self.action_realign_idx]
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, obs):
        obs = obs[self.obs_realign_idx] # realign observation into different cases
        return obs