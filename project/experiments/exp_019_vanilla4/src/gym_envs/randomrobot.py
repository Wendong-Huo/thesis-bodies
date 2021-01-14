import numpy as np
import pybullet
from gym_envs.my_envs import MyWalkerBase, MyWalkerBaseBulletEnv

class MyRandomRobot(MyWalkerBase):
    foot_list = []

    def __init__(self, xml):
        with open(xml, "r") as f:
            lines = f.readlines()
        joint_number = 0
        for line in lines:
            if line.strip().startswith("joint_number:"):
                joint_number = int(line.split(":")[-1].strip())
        assert joint_number!=0
        print(f"joint_number: {joint_number}")
        super().__init__(xml, "torso", action_dim=joint_number, obs_dim=8+joint_number*2, power=0.10)

    def alive_bonus(self, z, pitch):
        if z>self.initial_z*3: # avoid fly-away bug
            print("Fly-away Bug!")
            exit(1)
        # straight = z > 0.2 # torso length is 1.0
        # return +1 if straight else -1
        return 0

    def robot_specific_reset(self, bullet_client):
        super().robot_specific_reset(bullet_client)
        # bullet_client.setGravity(0,0,-1) # experiment in low gravity because of the pybullet "fly-away" bug
        bullet_client.changeVisualShape(1,-1,rgbaColor=[0.7,0.7,0.2,1.0]) # change torso color
        bullet_client.changeDynamics(0,-1,lateralFriction=0.999) # increase floor friction
        # for j in self.ordered_joints:
            # j
        # robot_aabb = bullet_client.getAABB(1)
        # height = robot_aabb[0][2]
        # print(f"height:", height)
        # height = -height + 1
        # bullet_client.resetBasePositionAndOrientation(1,[0,0,height],[0,0,0,1])



class MyRandomRobotEnv(MyWalkerBaseBulletEnv):

    def __init__(self, xml, render=False):
        self.robot = MyRandomRobot(xml=xml)
        self.xml = xml
        super().__init__(self.robot, render)

