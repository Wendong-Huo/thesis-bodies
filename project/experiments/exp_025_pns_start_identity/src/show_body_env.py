import time
import glob
from common import gym_interface
import pybullet as p
import os
import pybullet_data
import gym
import pybullet_envs
import shutil
import re
import numpy as np
import random

env = gym_interface.make_env(robot_body=207, render=True, dataset_folder="output_data/bodies")()
obs = env.reset()
env.env._p.setGravity(0,0,-1)
a = env.action_space.sample()

for i, j in enumerate(env.robot.ordered_joints):
    print(i, j.jointIndex, j.joint_name)

a = np.zeros_like(a)

while True:
    # a[2] = (random.random()-0.5)*10
    a[6] = (random.random()-0.5)
    env.step(a)
    time.sleep(0.01)