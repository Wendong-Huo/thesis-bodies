import numpy as np
import pybullet
from gym_envs.my_envs import MyWalkerBase, MyWalkerBaseBulletEnv


class MyHopper(MyWalkerBase):
    foot_list = ["foot"]

    def __init__(self, xml):
        super().__init__(xml, "torso", action_dim=3, obs_dim=15, power=0.75)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1


class MyHopperBulletEnv(MyWalkerBaseBulletEnv):

    def __init__(self, xml, render=False):
        self.robot = MyHopper(xml=xml)
        self.xml = xml
        super().__init__(self.robot, render)
