import argparse, os
from stable_baselines3.common.utils import constant_fn, set_random_seed
import random
import numpy as np
import gym
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
import pybullet_envs
import torch
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv
import pybullet as p

sysprint = print
g_line = 0
def print(*args):
    global g_line
    CSI="\x1B["
    sysprint(CSI+"31;40m\n"+f"{g_line}> ", end="")
    sysprint(*args)
    sysprint(CSI + "0m")
    g_line+=1
def printt(*args):
    global g_line
    CSI="\x1B["
    sysprint(CSI+"32;40m\n"+f"{g_line}> ", end="")
    sysprint(*args)
    sysprint(CSI + "0m")
    g_line+=1

if __name__ == "__main__":  # noqa: C901
    torch.manual_seed(0)
    torch.set_deterministic(True)

    os.environ["CUDA_VISIBLE_DEVICES"]=""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_random_seed(args.seed)
    print(f"Seed: {args.seed}")
    print(f"random: {random.random()}")
    print(f"np.random: {np.random.random()}")
    k = torch.rand([10])
    print(k)


    env = gym.make("Walker2DBulletEnv-v0")
    env.seed(args.seed)
    env.reset()

    printt("joint position")
    print(env.env.robot.jdict["thigh_joint"].get_state())

    printt("before 100 steps")
    print(env.env.robot.body_real_xyz)
    
    env.action_space.seed(args.seed)
    printt("action space sample")
    print(env.action_space.sample())
    env.step(env.action_space.sample())
    printt("after 1 step")
    print(env.env.robot.jdict["thigh_joint"].get_state())


    for i in range(1000):
        env.step(env.action_space.sample())
    printt("after 1000 steps")
    print(env.env.robot.body_real_xyz)
    print(env.env.robot.jdict["thigh_joint"].get_state())
    env.reset()
    for i in range(5):
        obs, r, done, info = env.step(env.action_space.sample())  
        if done:
            printt(i)
            break
    printt("after reset and 9 steps")
    print(env.env.robot.body_real_xyz)
    print(env.env.robot.jdict["thigh_joint"].get_state())

    print(env.robot.np_random.uniform(low=-0.1, high=0.1))
    model = PPO(policy="MlpPolicy", env=env, seed=args.seed)
    printt("before train:")
    weight_sum = 0
    for param in model.policy.parameters():
        weight_sum += np.sum(param.data.numpy(), keepdims=False)
        # print(param.data, param.size())
    print(weight_sum)

    model.learn(total_timesteps=2000)
    printt("after train:")
    weight_sum = 0
    for param in model.policy.parameters():
        weight_sum += np.sum(param.data.numpy(), keepdims=False)
        # print(param.data, param.size())
    print(weight_sum)
