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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from common import wrapper_custom_align
from common import common

if __name__ == "__main__":
    wrappers = [wrapper_custom_align.CustomAlignWrapper]
    
    _envs = []
    for i in range(len(common.args.train_bodies)):
        _envs.append(gym_interface.make_env(rank=i, wrappers=wrappers,robot_body=common.args.train_bodies[i], render=True, force_render=True))

    env = SubprocVecEnv(_envs)
    obs = env.reset()
    # env.env._p.setGravity(0,0,-1)
    a = env.action_space.sample()

    # for i, j in enumerate(env.robot.ordered_joints):
    #     print(i, j.jointIndex, j.joint_name)

    a = np.zeros_like(a)

    history_actions = []
    while True:
        # a[2] = (random.random()-0.5)*10
        # a[6] = (random.random()-0.5)
        a = env.action_space.sample()
        history_actions.append(np.mean(np.abs(a)))
        if len(history_actions)>100:
            print(f"Avg action value {np.mean(history_actions)}")
            history_actions=[]
        a = np.stack([a]*env.num_envs, axis=0)
        env.step(a)
        time.sleep(0.01)