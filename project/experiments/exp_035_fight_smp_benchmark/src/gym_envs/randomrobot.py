import numpy as np
import pybullet
from gym_envs.my_envs import MyWalkerBase, MyWalkerBaseBulletEnv

from common import colors
from common import common
class MyRandomRobot(MyWalkerBase):
    foot_list = []

    def __init__(self, xml, num_joints=-1, self_collision=False, power=0.1):
        if num_joints==-1: # Read number of joints from xml file
            with open(xml, "r") as f:
                lines = f.readlines()
            joint_number = 0
            for line in lines:
                if line.strip().startswith("joint_number:"):
                    joint_number = int(line.split(":")[-1].strip())
            assert joint_number!=0
            print(f"joint_number: {joint_number}")
        else:
            joint_number = num_joints

        super().__init__(xml, "torso", action_dim=joint_number, obs_dim=8+joint_number*2, power=power)
        self.r_self_collision = self_collision
        self.self_collision_enabled = False
        self.colored = False

    def alive_bonus(self, z, pitch):
        if z>self.initial_z*3: # avoid fly-away bug
            print("Fly-away Bug!")
            from datetime import datetime
            _dump = f"{datetime.now().ctime()} Related robot_id {self.robot_id}\n{common.args}\nDumps {locals()}\n"
            print(_dump)
            with open("output_data/tmp/FlyawayBug.txt", "a") as f:
                print(_dump, file=f)
            exit(1)
        # straight = z > 0.2 # torso length is 1.0
        # return +1 if straight else -1
        return 0

    def robot_specific_reset(self, bullet_client):
        super().robot_specific_reset(bullet_client)
        # bullet_client.setGravity(0,0,-1) # experiment in low gravity because of the pybullet "fly-away" bug
        bullet_client.changeDynamics(0,-1,lateralFriction=0.999) # increase floor friction
        if self.r_self_collision and not self.self_collision_enabled:
            self.set_self_collision(bullet_client)
            self.self_collision_enabled = True

    def set_self_collision(self, bullet_client):
        """need to do this manually: https://github.com/bulletphysics/bullet3/issues/1740"""
        all_link_ids = [-1]
        all_parent_relationship = {}
        for j in self.ordered_joints:
            ret = bullet_client.getJointInfo(1,j.jointIndex)
            # print(ret)
            # print(f"joint index: {ret[0]}, link name: {ret[-5]}, parent index: {ret[-1]})")
            linkIndex = ret[0]+1 # TODO: Is this always the case? linkId = jointId+1

            all_link_ids.append(linkIndex)
            all_parent_relationship[linkIndex] = ret[-1]
        
        for j1 in all_link_ids:
            for j2 in all_link_ids:
                if j1<=j2: # only check each pair once
                    continue
                if all_parent_relationship[j1]==j2: # equivalent to URDF_USE_SELF_COLLISION_INCLUDE_PARENT
                    continue
                # print(f"set self collision pair: {j1},{j2}")
                bullet_client.setCollisionFilterPair(1,1,j1,j2,1)

class MyRandomRobotEnv(MyWalkerBaseBulletEnv):

    def __init__(self, xml, num_joints=-1, self_collision=True, power=0.1, render=False):
        # Warning: Enable self collision will increase the possibility of the "Fly-away" bug!
        #          The body plan will perform differently if this attribute is different, as if it is a completely new body plan. Need to re-search for workable bodies.
        self.robot = MyRandomRobot(xml=xml, num_joints=num_joints, self_collision=self_collision, power=power)
        self.xml = xml
        super().__init__(self.robot, render)

    def _reset_color(self):
        self.pybullet.changeVisualShape(1, -1, rgbaColor=[0.3, 0.3, 0.3, 1.0]) # change torso color
        super()._reset_color()