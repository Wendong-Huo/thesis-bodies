import glob, pickle, os
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_envs.scene_abstract import World
from PyBulletWrapper.pybullet_wrapper.base import BaseWrapperPyBullet
from PyBulletWrapper.pybullet_wrapper.handy import HandyPyBullet
import yaml

p = BaseWrapperPyBullet(p)
p = HandyPyBullet(p)


# Set up pybullet
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)
p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=-10, cameraPitch=-10, cameraTargetPosition=[2, 0, 0])
# this is the floor used by pybullet_envs, other floors such as plane.urdf might be slipper.
filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
ground_plane_mjcf = p.loadSDF(filename)
for i in ground_plane_mjcf:
    p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
    p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
World(p,gravity=9.8, timestep=0.0165 / 4, frame_skip=4)

# Load bodies
pos = 0
mutants = []
xmls = glob.glob("./bodies/*.xml")
for i, xml in enumerate(xmls):
    mutant, = p.loadMJCF(xml)
    mutants.append(mutant)
    p.resetBasePositionAndOrientation(bodyUniqueId=mutant, posObj=[0,pos*2,2], ornObj=[0,0,0,1])
    pos += 1
    with open(f"./params/{i}.yaml", "r") as f:
        params_float = yaml.load(f)
    print(f"{i}: {len(params_float)}")
    print(params_float)


while True:
    for mutant in mutants:
        total_j = p.getNumJointsPy(mutant)
        for j in range(total_j):
        # j = 4
            joint = p.getJointInfoPy(bodyUniqueId=mutant, jointIndex=j)
            # print(joint["jointName"])
            # print(f"set joint {j}")
            p.setJointMotorControl2(mutant, jointIndex=j, controlMode=p.TORQUE_CONTROL, force=(np.random.random()-0.5)*200)
    p.stepSimulation()
    p.sleepPy(0.01)