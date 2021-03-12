import numpy as np
import pybullet
from gym_envs.my_envs import MyWalkerBase, MyWalkerBaseBulletEnv


class MyHalfCheetah(MyWalkerBase):
    foot_list = ["ffoot", "fshin", "fthigh", "bfoot", "bshin", "bthigh"]  # track these contacts with ground

    def __init__(self, xml):
        super().__init__(xml, "torso", action_dim=6, obs_dim=26, power=0.90)

    def alive_bonus(self, z, pitch):
        # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
        return +1 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[2] \
            and not self.feet_contact[4] and not self.feet_contact[5] else -1

    def robot_specific_reset(self, bullet_client):
        super().robot_specific_reset(bullet_client)
        self.jdict["bthigh"].power_coef = 120.0
        self.jdict["bshin"].power_coef = 90.0
        self.jdict["bfoot"].power_coef = 60.0
        self.jdict["fthigh"].power_coef = 140.0
        self.jdict["fshin"].power_coef = 60.0
        self.jdict["ffoot"].power_coef = 30.0


class MyHalfCheetahBulletEnv(MyWalkerBaseBulletEnv):

    def __init__(self, xml, render=False):
        self.robot = MyHalfCheetah(xml=xml)
        self.xml = xml
        super().__init__(self.robot, render)

    def set_collision(self):
        # enable special collision:
        raise NotImplemented # PyBullet issue, setCollisionFilterPair doesn't work.
