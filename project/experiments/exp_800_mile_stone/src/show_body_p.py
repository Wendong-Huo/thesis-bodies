from pickle import load
import time
import glob
import pybullet as p
import os
import pybullet_data
import shutil
import re
import numpy as np
import random
import yaml
import cv2

from common import gym_interface, linux
from common import colors

def _reset_link_color(pybullet, robot_id, realign_idx):
    color_idx = 0
    joints = []
    for joint_id in range(pybullet.getNumJoints(robot_id)):

        j = pybullet.getJointInfo(robot_id, joint_id)
        joints.append({"index": j[0], "jointName":j[1], "linkName":j[12]})
    print(joints)

    pybullet.changeVisualShape(robot_id, -1,rgbaColor=[0.3, 0.3, 0.3, 1.0]) # change torso color
    for joint in joints:
        linkName = joint["linkName"].decode('utf8')
        if linkName.startswith("link0_"):
            continue
        if linkName.startswith("torso") or linkName.startswith("aux_"):
            pybullet.changeVisualShape(robot_id, joint["index"],rgbaColor=[0.3, 0.3, 0.3, 1.0]) # change color
        else:
            print(linkName, joint["index"], color_idx)
            pybullet.changeVisualShape(robot_id, joint["index"],rgbaColor=colors.get_link_color(realign_idx[color_idx])) # change color
            # pybullet.changeVisualShape(robot_id, joint["index"],rgbaColor=[0.3, 0.3, 0.3, 1.0]) # change color
            color_idx += 1
    # for joint in joints:
    #     print(linkName, joint["index"], color_idx)
    #     pybullet.changeVisualShape(robot_id, joint["index"], rgbaColor=colors.get_link_color(1))
def set_torque(jointIndex, torque):
    p.setJointMotorControl2(bodyIndex=robot,
                                jointIndex=jointIndex,
                                controlMode=p.TORQUE_CONTROL,
                                force=torque)

p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(flag=p.COV_ENABLE_MOUSE_PICKING, enable=1,lightPosition=[10,-10,10])
p.resetDebugVisualizerCamera(3, 0, -20, [0,0,0.5])
linux.fullscreen()

M2 = "0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,3,4,5,6,0,7::4,0,2,3,1,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7"
M0 = "0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7"
M32 = "4,3,2,1,7,5,6,0::1,2,4,5,0,3,6,7::1,2,3,7,6,5,4,0::1,2,0,4,5,3,6,7::2,1,4,3,0,6,5,7::2,1,3,5,0,4,6,7::2,1,0,4,5,3,6,7::0,1,4,3,6,7,2,5"
M4 = "5,1,2,3,4,0,6,7::1,2,0,5,4,3,6,7::1,2,3,0,4,5,6,7::1,2,4,3,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7"
TWO_ROBOTS_1 = "0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7"
TWO_ROBOTS_2 = "0,1,2,3,4,5,6,7::7,6,5,4,3,2,1,0"
NoColor = "0,0,0,0,0,0,0,0::0,0,0,0,0,0,0,0"
arrangement = NoColor.split("::")
arrangement = [x.split(",") for x in arrangement]
for body in [500,900]:
    p.resetDebugVisualizerCamera(3 if gym_interface.template(body)=='ant' or gym_interface.template(body)=='walkerarms' else 2, 0, -20, [0,0,0.5])

    p.resetSimulation()
    time.sleep(0.1)
    filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
    (floor,) = p.loadSDF(filename)
    filename = f"../input_data/bodies/{body}.xml"
    (robot,) = p.loadMJCF(filename)
    p.removeBody(floor)

    print(f"\n\nbody {body}\n\n")
    
    _reset_link_color(p, robot, arrangement[0 if body==500 else 1])

    joint_names = {}
    for joint_id in range(p.getNumJoints(robot)):
        info = list(p.getJointInfo(robot, joint_id))
        info[1] = info[1].decode()
        print(f"joint {joint_id}", end=", ")
        print(info[1], end=", ")
        joint_names[info[1]] = joint_id
        d = p.getDynamicsInfo(robot, joint_id)
        print(d[0])


    while True:
        with open("../input_data/position4photos/positions.yml", "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        data = data[gym_interface.template(body)]

        joint_positions = p.getJointStates(robot, list(range(p.getNumJoints(robot))))
        joint_positions = np.zeros(p.getNumJoints(robot))
        
        if "position" in data:
            robot_position = [0,0,0]
            for key in data["position"]:
                robot_position[key] = data["position"][key]
            p.resetBasePositionAndOrientation(robot, robot_position, [0,0,0,1])

        if "joints" in data:
            for key in data["joints"]:
                joint_positions[key] = data["joints"][key]

        if "joints_name" in data:
            for key in data["joints_name"]:
                joint_positions[joint_names[key]] = data["joints_name"][key]

        for joint_id in range(p.getNumJoints(robot)):
            if joint_positions[joint_id]!=0:
                p.resetJointState(robot, joint_id, targetValue=joint_positions[joint_id])
        
        # for step in range(100000):
        #     set_torque(0,100)
        #     # break
        # p.stepSimulation()
        time.sleep(0.1)
        if True:
            (width, height, rgbPixels, _, _) = p.getCameraImage(1920,1080, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            image = rgbPixels[:,:,:3]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"output_data/saved_images/{gym_interface.template(body)}_{body}_nocolor.png", image)
            break  