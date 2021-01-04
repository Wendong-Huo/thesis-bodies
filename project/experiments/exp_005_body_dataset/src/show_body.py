import time
import common.gym_interface as gym_interface
import pybullet as p
import os
import pybullet_data
import shutil

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
for i in range(1000000):
    p.resetSimulation()
    filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
    _ = p.loadSDF(filename)
    from_filename = "/home/liusida/thesis-bodies/project/experiments/exp_005_body_dataset/input_data/body-templates/600.xml"
    to_filename = f"/tmp/{i}.xml"
    shutil.copy(from_filename, to_filename)
    p.loadMJCF(to_filename)
    time.sleep(3)
# env = gym_interface.make_env(rank=0, render=True, robot_body=401, dataset_folder="../input_data/body-templates")()

# env.reset()
    # # for i in range(1000):
    # #     action = env.action_space.sample()
    # #     obs, reward, done, _ = env.step(action)
    # #     time.sleep(100)

    # time.sleep(3)
    # env.close()