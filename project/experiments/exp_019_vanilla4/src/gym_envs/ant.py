import numpy as np
import pybullet
from gym_envs.my_envs import MyWalkerBase, MyWalkerBaseBulletEnv


class MyAnt(MyWalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self, xml):
        super().__init__(xml, "torso", action_dim=8, obs_dim=28, power=2.5)

    def alive_bonus(self, z, pitch):
        # I can't remember why I changed this: 
        # return +1 if self.initial_z-0.49 < z < self.initial_z*2 else -1  # self.initial_z-0.5 is central sphere rad, die if it scrapes the ground
        return +1 if z > self.initial_z * 0.347 else -1  # self.initial_z-0.5 is central sphere rad, die if it scrapes the ground
        # for vanilla Ant, self.initial_z * 0.347 = 0.26


class MyAntBulletEnv(MyWalkerBaseBulletEnv):

    def __init__(self, xml, render=False):
        self.robot = MyAnt(xml=xml)
        self.xml = xml
        super().__init__(self.robot, render)
