import os
import numpy as np
import gym
import pybullet_envs
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
def make_env(env_id, rank=0, seed=0, log_dir=None, wrapper_class=None, env_kwargs=None):
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
    if env_kwargs is None:
        env_kwargs = {}
    def _init():
        env = gym.make(env_id, **env_kwargs)
        env.seed(seed * 128 + rank)
        return env
    return _init
env = DummyVecEnv(
    [
        make_env("Walker2DBulletEnv-v0", 0, 0, log_dir="tmp")
    ]
)
env = VecNormalize.load("tmp/vecnormalize1.pkl", env)
env.training = False
print(dir(env))
env.reset()
for i in range(10):
    env.step([env.action_space.sample()])
    
obs = env.step([env.action_space.sample()])
original_obs = env.get_original_obs()
print(obs)
print(original_obs)