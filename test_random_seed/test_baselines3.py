import argparse, os
from stable_baselines3.common.utils import constant_fn, set_random_seed
import numpy as np
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
import pybullet_envs
import gym

if __name__ == "__main__":  # noqa: C901
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    env = gym.make("Walker2DBulletEnv-v0")
    # env.seed(0)
    model = PPO(policy="MlpPolicy", env=env, seed=0)
    model.learn(total_timesteps=10000)
    print("after train:")
    weight_sum = 0
    for param in model.policy.parameters():
        weight_sum += np.sum(param.data.numpy(), keepdims=False)
        # print(param.data, param.size())
    print(weight_sum)
