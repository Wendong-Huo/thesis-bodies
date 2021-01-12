import numpy as np
import pybullet
from gym_envs.my_envs import MyWalkerBase, MyWalkerBaseBulletEnv

class MyWalkerArms(MyWalkerBase):
    foot_list = ["foot", "foot_left"]

    def __init__(self, xml):
        super().__init__(xml, "torso", action_dim=8, obs_dim=26, power=0.40)

    def alive_bonus(self, z, pitch):
        straight = z > self.initial_z * 0.64 and abs(pitch) < 1.0
        return +1 if straight else -1

    def robot_specific_reset(self, bullet_client):
        super().robot_specific_reset(bullet_client)
        for n in ["foot_joint", "foot_left_joint"]:
            self.jdict[n].power_coef = 30.0


class MyWalkerArmsBulletEnv(MyWalkerBaseBulletEnv):

    def __init__(self, xml, render=False):
        self.robot = MyWalkerArms(xml=xml)
        self.xml = xml
        super().__init__(self.robot, render)

