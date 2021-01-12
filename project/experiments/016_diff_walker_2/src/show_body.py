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

files = glob.glob("output_data/bodies/*.xml")
files = ["output_data/bodies/800.xml"]

for bodyfile in files:
    # if "303" not in bodyfile:
    #     continue
    print(bodyfile)
    m = re.findall(r"\/([0-9]+)\.xml", bodyfile)
    assert len(m)==1, f"Filename not right. {bodyfile}"
    body = m[0]
    print(f"body {body}")
    p.resetSimulation()
    filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
    _ = p.loadSDF(filename)
    from_filename = "/home/liusida/thesis-bodies/project/experiments/exp_005_body_dataset/input_data/body-templates/300.xml"
    from_filename = f"../input_data/bodies/{body}.xml"
    # to_filename = f"/tmp/{i}.xml"
    to_filename = from_filename
    # shutil.copy(from_filename, to_filename)
    (robot,) = p.loadMJCF(to_filename)
    print(f"\n\nbody {body}\n\n")
    for joint_id in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, joint_id)
        print(f"joint {joint_id}", end=", ")
        print(info[1], end=", ")
        d = p.getDynamicsInfo(robot, joint_id)
        print(d[0])
    
    time.sleep(20)
