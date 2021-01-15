# Wrappers for 900 mutants, different topology, same dimension

import sys
import gym
import numpy as np

from common import common
from common import gym_interface
from common import colors
class CustomAlignWrapper(gym.ObservationWrapper):
    max_num_joints = 8

    """For body 200s, different topology, different observation space, at most 8 joints, define alignment in arguments."""
    def __init__(self, env):
        env.foot_list = []
        super().__init__(env)
        self.debug = 1
        self.colored = False
        # assert gym_interface.template(env.robot.robot_id)=="randomrobot", "CustomAlignWrapper only support randomrobot for now."
        # obs ------------ F() ------> abs_obs
        # abs_action --- F^{-1}() ---> action
        self.parse_alignment(common.args.custom_alignment) # a string contain (16-1)x8, in the format of e.g. "0,1,2::2,1,0::2,0,1"
        self.action_realign_idx = self.realign_idx
        self.obs_realign_idx = list(range(8))
        for i in self.realign_idx:
            self.obs_realign_idx.append(8+i*2)
            self.obs_realign_idx.append(8+i*2+1)
        self.obs_realign_idx = np.argsort(self.obs_realign_idx).tolist()
        print(self.robot.robot_id)
        print(self.obs_realign_idx)

        self.num_null_joints = 0
        self.original_num_joints = env.action_space.shape[0]
        assert self.original_num_joints <= self.max_num_joints, f"CustomAlignWrapper only support at most {self.max_num_joints}-joint body."
        # always reset the space size, because it removes the foot_contact part
        self.reset_space_size()
    
    def reset(self):
        obs = super().reset()
        self.reset_link_color()
        return obs


    def reset_link_color(self):
        if self.isRender and not self.colored and hasattr(self, "pybullet"):
            self.pybullet.configureDebugVisualizer(flag=self.pybullet.COV_ENABLE_MOUSE_PICKING, enable=0,lightPosition=[10,-10,10])

            color_idx = 0
            for part_name, part in self.robot.parts.items():
                if part_name.startswith("link0_") or part_name.startswith("floor") or part_name.startswith("aux_"):
                    continue
                if part_name.startswith("torso"):
                    self.pybullet.changeVisualShape(1,part.bodyPartIndex,rgbaColor=[0.3, 0.3, 0.3, 1.0]) # change color
                else:
                    print(part_name, part.bodyPartIndex, color_idx)
                    self.pybullet.changeVisualShape(1,part.bodyPartIndex,rgbaColor=colors.get_link_color(self.realign_idx[color_idx])) # change color
                    color_idx += 1
            self.colored = True

    def reset_space_size(self):
        # re-calculate observation space size
        self.num_null_joints = self.max_num_joints - self.action_space.shape[0]
        self.old_obs_space = self.observation_space
        low = self.old_obs_space.low[0]
        high = self.old_obs_space.high[0]
        self.obs_length = 8 + self.max_num_joints*2 # foot contact removed for now
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=[self.obs_length])
        # re-calculate action space size
        self.old_action_space = self.action_space
        low = self.old_action_space.low[0]
        high = self.old_action_space.high[0]
        self.action_space = gym.spaces.Box(low=low, high=high, shape=[self.max_num_joints])

    def step(self, action):
        if self.debug:
            print(self.robot.robot_id)
            print(self.action_realign_idx)
            print(self.obs_realign_idx)
            self.debug=0
        # re-align F^{-1}()
        action = action[self.action_realign_idx]
        # shorten
        action = action[:self.original_num_joints]
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, obs):
        # expand
        if self.num_null_joints>0:
            obs = np.concatenate((
                obs,
                [0] * (self.num_null_joints*2),
            ))
        # re-align F()
        obs = obs[self.obs_realign_idx] # realign observation into different cases
        return obs

    def parse_alignment(self, custom_alignment_string):
        alignment = None
        alignments = custom_alignment_string.split("::")
        assert len(alignments)==common.args.num_venvs, f"Not enough alignment for {common.args.num_venvs} envs."
        for i,a in enumerate(alignments):
            if self.env.rank == i:
                b = a.split(",")
                assert len(b)==self.max_num_joints, f"CustomAlignWrapper only support an observation space size of {self.max_num_joints}"
                alignment = [int(x.strip()) for x in b]
        assert alignment is not None, f"Alignment for rank {self.rank} is not found."
        self.realign_idx = alignment
        print(f"Joint order for {self.robot.robot_id}")
        print(alignment)

