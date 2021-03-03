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
from common.utils import linux_fullscreen

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

for body in range(900,916):
    p.resetDebugVisualizerCamera(3 if gym_interface.template(body)=='ant' or gym_interface.template(body)=='walkerarms' else 2, 0, -20, [0,0,0.5])

    p.resetSimulation()
    time.sleep(0.1)
    filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
    (floor,) = p.loadSDF(filename)
    filename = f"../input_data/bodies/{body}.xml"
    (robot,) = p.loadMJCF(filename)
    p.removeBody(floor)

    print(f"\n\nbody {body}\n\n")
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
            cv2.imwrite(f"output_data/saved_images/{gym_interface.template(body)}_{body}.png", image)
            break  