import time
import glob
import common.gym_interface as gym_interface
import pybullet as p
import os
import pybullet_data
import shutil
import re
import numpy as np
import random
def set_torque(jointIndex, torque):
    p.setJointMotorControl2(bodyIndex=robot,
                                jointIndex=jointIndex,
                                controlMode=p.TORQUE_CONTROL,
                                force=torque)

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
floor=False
for body in [399]:
    print(f"body {body}")
    p.resetSimulation()
    if floor:
        filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
        _ = p.loadSDF(filename)
    filename = f"../input_data/bodies/{body}.xml"
    (robot,) = p.loadMJCF(filename)
    print(f"\n\nbody {body}\n\n")
    for joint_id in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, joint_id)
        print(f"joint {joint_id}", end=", ")
        print(info[1], end=", ")
        d = p.getDynamicsInfo(robot, joint_id)
        print(d[0])
    
    for step in range(100000):
        set_torque(0,100)
        # break
        p.stepSimulation()
        time.sleep(0.01)
