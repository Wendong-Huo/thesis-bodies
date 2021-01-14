# Wrappers for 900 mutants, different topology, same dimension

import sys
import gym
import numpy as np

from common import common
from common import gym_interface
from common import colors

class MutantWrapper(gym.ObservationWrapper):
    """Mutants 900, different topology, but same dimension for simplicity."""
    def __init__(self, env):
        super().__init__(env)
        self.debug = 1
        self.colored = False
        self.enable_mutant_wrapper = True
        if common.args.disable_reordering: # This is for test time.
            self.enable_mutant_wrapper = False
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
        self.realign_idx = self.action_realign_idx
        
        if not self.enable_mutant_wrapper:
            # reset everything
            self.realign_idx = sorted(self.realign_idx)
            self.action_realign_idx = sorted(self.action_realign_idx)
            self.obs_realign_idx = sorted(self.obs_realign_idx)


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
