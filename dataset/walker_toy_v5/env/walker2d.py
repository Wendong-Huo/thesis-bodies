# This is a generated file!
# walker2d.py is generated from walker2d.template
# 
import numpy as np
import pybullet
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase
from pybullet_envs.scene_stadium import MultiplayerStadiumScene
import pybullet_data

from pathlib import Path

class Walker2D(WalkerBase):
    foot_list = ['foot', 'foot_left']
    def __init__(self, xml, param):
        self.param = param
        WalkerBase.__init__(self, xml, "torso", action_dim = 6 , obs_dim= 22 , power=0.5)
        self.height_using_knees = (self.param["z0"] + self.param["z1"]) / 2 - self.param["z2"] + 0.1
        # TODO: adjust power according to how heavy the parts are. 
        # Too large power will cause simulation numerical error.
        # Too small power will cause the agent hard to move a step.
        # Also, action_dim and obs_dim need to change according to how many parts are there.
        # obs_dim = 8 + action_dim * 2 + len(foot_list)

    def alive_bonus(self, z, pitch):
        # return +1 if z > self.height_using_knees and abs(pitch) < 1.0 else -1
        return 0

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        
        # power coefficient should be proportional to the min possible volume of that part. (Avoid pybullet fly-away bug.)
        self.jdict["thigh_joint"].power_coef = 65;self.jdict["leg_joint"].power_coef = 31;self.jdict["foot_joint"].power_coef = 18;self.jdict["thigh_left_joint"].power_coef = 65;self.jdict["leg_left_joint"].power_coef = 31;self.jdict["foot_left_joint"].power_coef = 18

        # I deleted ignore_joints in mujoco xml files, so i need to place the robot at an appropriate initial place manually.
        robot_id = self.objects[0] #  is the robot pybullet_id
        bullet_client.resetBasePositionAndOrientation(bodyUniqueId=robot_id, posObj=[0,0,self.param["torso_center_height"]+0.1], ornObj=[0,0,0,1]) # Lift the robot higher above ground


class Walker2DEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False, debug=False, xml="", max_episode_steps=1000, is_eval=False, param=[]):
        """ param is a ndarray stores the parameters of the body """
        self.robot = Walker2D(xml, param)
        self.debug = debug
        self.step_num = 0
        self.max_episode_steps = max_episode_steps
        self.is_eval = is_eval
        self.param = param
        self.forbid_links = [-1]
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

    def _isDone(self):
        return False

    def reset(self):
        self.step_num = 0
        obs = super().reset()

        if self.isRender:
            self._p.changeVisualShape(objectUniqueId=1, linkIndex=-1, rgbaColor=[1.0, 0.8, 0.4, 1.0]) # turn "head" golden

        self.forbid_links = [-1] # if "torso" touches the ground
        num_joint = self._p.getNumJoints(bodyUniqueId=self.robot.objects[0])
        for i in range(num_joint):
            joint = self._p.getJointInfo(bodyUniqueId=self.robot.objects[0], jointIndex=i)
            name = joint[12] # joint["linkName"]
            if name==b"thigh" or name==b"thigh_left":
                self.forbid_links.append(i)

        return obs

    def forbid_links_touch_ground(self):
        pts = self._p.getContactPoints(bodyA=1,bodyB=0) #-1 is "head"
        for pt in pts:
            for forbid in self.forbid_links:
                if pt[3]==forbid: # pt["linkA"]
                    return True

    def step(self, a):
        self.step_num += 1
        if np.mean(np.abs(a))>0.9:
            print("Warining! Action close to 1.0!")
        obs, r, done, info = super().step(a)
        if self.forbid_links_touch_ground():
            done = True
        if self.step_num > self.max_episode_steps:
            done = True
        if done:
            r -= 10. # explicitly punish falling down
        return obs, r, done, info

