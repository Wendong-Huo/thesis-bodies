import time
import glob
import common.gym_interface as gym_interface
import pybullet as p
import os
import pybullet_data
import shutil
import re
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
files = glob.glob("../input_data/bodies/900.xml")
files.sort()
print(files)
p.resetSimulation()

filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
_ = p.loadSDF(filename)
robots = [None]*len(files)
for i in range(len(files)):
    (robots[i],) = p.loadMJCF(files[i])
while True:
    p.stepSimulation()
    time.sleep(0.01)