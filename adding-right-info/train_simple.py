import os
import yaml
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

import utils

if __name__ == "__main__":  # noqa: C901
    folder = utils.folder
    os.makedirs(folder, exist_ok=True)

    hyperparams = utils.load_hyperparameters()

    normalize_kwargs = {}
    normalize_kwargs["gamma"] = hyperparams["gamma"]
    
    args = utils.args
    
    # PPO.learn need this. If use SubprocVecEnv instead of DummyVecEnv, you need to seed in each subprocess.
    set_random_seed(utils.seed)

    debug = args.debug
    train_on_both_bodies = args.train_on_both_bodies
    with_bodyinfo = args.with_bodyinfo
    train_num_envs = 16 if not debug else 2
    total_timesteps = 5e6 if not debug else 1
    
    
    if train_on_both_bodies:
        training_bodies = args.train_bodies
        print(training_bodies)
        if with_bodyinfo:
            env = DummyVecEnv([utils.make_env(rank=i, seed=utils.seed, render=args.render, robot_body=training_bodies[i%2], body_info=training_bodies[i%2]) for i in range(train_num_envs)])
            save_filename = f"model-ant-{training_bodies[0]}-{training_bodies[1]}-with-bodyinfo"
        else:
            env = DummyVecEnv([utils.make_env(rank=i, seed=utils.seed, render=args.render, robot_body=training_bodies[i%2], body_info=0) for i in range(train_num_envs)])
            save_filename = f"model-ant-{training_bodies[0]}-{training_bodies[1]}"
    else:
        body = args.body_id
        print(body)
        env = DummyVecEnv([utils.make_env(rank=i, seed=utils.seed, render=args.render, robot_body=body, body_info=0) for i in range(train_num_envs)])
        save_filename = f"model-ant-single-{body}"

    env = VecNormalize(env, **normalize_kwargs)

    keys_remove =["normalize", "n_envs", "n_timesteps", "policy"]
    for key in keys_remove:
        del hyperparams[key]

    eval_env = DummyVecEnv([utils.make_env(rank=0, seed=utils.seed+1, render=False, robot_body=2, body_info=0)])
    eval_env = VecNormalize(eval_env, norm_reward=False, **normalize_kwargs)
    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=3,
        eval_freq=1e4, # will implicitly multiplied by 16 (train_num_envs)
        deterministic=True,
    )
    # eval_callback = None

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"{folder}/tb/{save_filename}", seed=utils.seed, **hyperparams)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{folder}/{save_filename}")
    # Important: save the running average, for testing the agent we need that normalization
    model.get_vec_normalize_env().save(f"{folder}/{save_filename}-vecnormalize.pkl")

    env.close()
