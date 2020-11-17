import argparse
import pickle
import importlib
import os
from time import sleep

import gym
import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.utils import StoreDict
from utils.wrappers import TimeFeatureWrapper

import load_dataset

def enjoy(stats_path, model_path, dataset, body_id, algo, n_timesteps=200, test_time=3, render=False, seed=0):
    dataset_name, env_id, train_files, train_params, train_names, test_files, test_params, test_names = load_dataset.load_dataset(
        dataset, seed=0, shuffle=False, train_proportion=1)
    set_random_seed(seed * 128 + 127)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)
    env_kwargs = {
        "xml": train_files[body_id],
        "param": train_params[body_id],
        "max_episode_steps": n_timesteps+1,
        "render": render,
    }
    env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=stats_path,
        seed=seed,
        log_dir="tmp/",
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    kwargs = dict(seed=seed)
    model = ALGOS[algo].load(model_path, env=env, **kwargs)
    obs = env.reset()
    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0

    body_x_record = []
    for _run in range(test_time):
        body_x = 0
        for _step in range(n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=True)
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            body_x = env.envs[0].robot.body_xyz[0]
            obs, reward, done, infos = env.step(action)
            episode_reward += reward[0]
            ep_len += 1
            if render:
                sleep(0.01)
            if done:
                break
        body_x_record.append(body_x)
        obs = env.reset()
    body_x_record = np.array(body_x_record)
    env.close()
    return body_x_record

def main():  # noqa: C901
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    test_body_id = 48
    record = enjoy(
        stats_path=f"logs/train_on_10/ppo1/Walker2Ds-v0_1/Walker2Ds-v0", 
        model_path=f"logs/train_on_10/ppo1/Walker2Ds-v0_1/best_model.zip", 
        dataset="dataset/walker2d_v6",
        algo="ppo1",
        body_id=test_body_id, 
        n_timesteps=1000, test_time=1, render=True)

    # record = enjoy(
    #     stats_path=f"experiment_results/logs_3x100/9_1/ppo/Walker2Ds-v0_1/Walker2Ds-v0", 
    #     model_path=f"experiment_results/logs_3x100/9_1/ppo/Walker2Ds-v0_1/best_model.zip", 
    #     dataset="dataset/walker2d_v6",
    #     algo="ppo",
    #     body_id=9, 
    #     n_timesteps=1000, test_time=1, render=True)

    # mean_record = np.mean(record)
    print(f"test on {test_body_id}")
    print(f"distances: {record}")


if __name__ == "__main__":
    main()
