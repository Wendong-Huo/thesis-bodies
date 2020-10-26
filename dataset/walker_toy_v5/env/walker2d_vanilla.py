# This is not 100% vanilla, because still we need to replace the xml of new body.
#
import numpy as np
import pybullet
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv, Walker2DBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase, Walker2D
from pybullet_envs.scene_stadium import MultiplayerStadiumScene
import pybullet_data

from pathlib import Path


class _Walker2D(Walker2D):
    def __init__(self, xml, param, render=False):
        self.param = param
        WalkerBase.__init__(self, xml, "torso", action_dim=6, obs_dim=22, power=0.40)

    def robot_specific_reset(self, bullet_client):
        super().robot_specific_reset(bullet_client)

        # power coefficient should be proportional to the min possible volume of that part. (Avoid pybullet fly-away bug.)
        self.jdict["thigh_joint"].power_coef = 65
        self.jdict["leg_joint"].power_coef = 31
        self.jdict["foot_joint"].power_coef = 18
        self.jdict["thigh_left_joint"].power_coef = 65
        self.jdict["leg_left_joint"].power_coef = 31
        self.jdict["foot_left_joint"].power_coef = 18

        # I deleted ignore_joints in mujoco xml files, so i need to place the robot at an appropriate initial place manually.
        robot_id = self.objects[0]  # is the robot pybullet_id
        bullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=robot_id, posObj=[0, 0, self.param["torso_center_height"] + 0.1],
            ornObj=[0, 0, 0, 1])  # Lift the robot higher above ground


class Walker2DEnv(Walker2DBulletEnv):
    def __init__(self, xml, param, render=False, max_episode_steps=1000):
        self.robot = _Walker2D(xml=xml, param=param)
        self.max_episode_steps = max_episode_steps
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

    def reset(self):
        self.step_num = 0
        return super().reset()

    def step(self, a):
        self.step_num += 1
        obs, r, done, info = super().step(a)
        if self.step_num > self.max_episode_steps:
            done = True
        return obs, r, done, info
