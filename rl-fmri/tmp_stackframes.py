import gym
import pybullet_envs
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
import common.utils as utils

utils.folder = "exp5"
venv = DummyVecEnv([utils.make_env(template=utils.template(400), robot_body=400, wrapper=None)])
venv = VecFrameStack(venv, 4)
obs = venv.reset()
print(obs.shape)