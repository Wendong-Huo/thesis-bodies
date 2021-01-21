import os
import gym
import pyrobotdesign as rd
import pyrobotdesign_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
def make_env(rank=0, seed=0, render=False):
    def _init():
        env = gym.make("RobotLocomotion-v0")
        if seed:
            env.seed(seed*100+rank)
            env.action_space.seed(seed*100 + rank)
        if render:
            env.render()
        return env
    return _init

def main():
    num_envs = 16
    env = DummyVecEnv([make_env(rank=i, render=False) for i in range(num_envs)])
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
                eval_env=eval_env,
                best_model_save_path="models",
                n_eval_episodes=2,
                eval_freq=int(1e5/num_envs),
                deterministic=True,
            )
    model = PPO("MlpPolicy", env, tensorboard_log="output_data/tensorboard/", verbose=True)
    model.learn(total_timesteps=1e6, callback=eval_callback)
    model.save("output_data/models/default")

if __name__ == "__main__":
    main()