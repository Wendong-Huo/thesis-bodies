#--exp, --seed, --train-bodies, --test-bodies  --with-bodyinfo
import os
import yaml
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from callbacks.eval import EvalCallback_with_prefix
from callbacks.fromzoo import SaveVecNormalizeCallback
import utils
import wrapper

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
    with_bodyinfo = args.with_bodyinfo
    train_num_envs = 16 if not debug else 2
    total_timesteps = 2e6 if not debug else 1

    training_bodies = [int(x) for x in args.train_bodies.split(",")]
    str_ids = "-".join(str(x) for x in training_bodies)
    test_bodies = [int(x) for x in args.test_bodies.split(",")]
    
    default_wrapper = wrapper.BodyinfoWrapper
    if args.disable_wrapper:
        default_wrapper = None

    if with_bodyinfo:
        env = DummyVecEnv([utils.make_env(rank=i, seed=utils.seed, wrapper=default_wrapper, render=args.render, robot_body=training_bodies[i%len(training_bodies)], body_info=training_bodies[i%len(training_bodies)]//100) for i in range(train_num_envs)])
        save_filename = f"model-ant-{str_ids}-with-bodyinfo"
    else:
        env = DummyVecEnv([utils.make_env(rank=i, seed=utils.seed, wrapper=default_wrapper, render=args.render, robot_body=training_bodies[i%len(training_bodies)], body_info=0) for i in range(train_num_envs)])
        save_filename = f"model-ant-{str_ids}"

    print(save_filename)

    env = VecNormalize(env, **normalize_kwargs)

    keys_remove =["normalize", "n_envs", "n_timesteps", "policy"]
    for key in keys_remove:
        del hyperparams[key]

    all_callbacks = []
    for test_body in test_bodies:
        if with_bodyinfo:
            if args.test_as_class<0:
                body_info = test_body//100
            else:
                body_info = args.test_as_class
        else:
            body_info = 0
        eval_env = DummyVecEnv([utils.make_env(rank=0, seed=utils.seed+1, wrapper=default_wrapper, render=False, robot_body=test_body, body_info=body_info)])
        eval_env = VecNormalize(eval_env, norm_reward=False, **normalize_kwargs)
        eval_callback = EvalCallback_with_prefix(
            eval_env=eval_env,
            prefix=f"{test_body}",
            n_eval_episodes=3,
            eval_freq=1e3, # will implicitly multiplied by (train_num_envs)
            deterministic=True,
        )
        all_callbacks.append(eval_callback)

    if args.with_checkpoint:
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=f'{folder}/checkpoints/', name_prefix=args.train_bodies)
        save_vec_callback = SaveVecNormalizeCallback(save_freq=1000, save_path=f"{folder}/checkpoints/", name_prefix=args.train_bodies)
        all_callbacks.append(checkpoint_callback)
        all_callbacks.append(save_vec_callback)

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"{folder}/tb/{save_filename}-s{utils.seed}", seed=utils.seed, **hyperparams)

    model.learn(total_timesteps=total_timesteps, callback=all_callbacks)
    model.save(f"{folder}/{save_filename}")
    # Important: save the running average, for testing the agent we need that normalization
    model.get_vec_normalize_env().save(f"{folder}/{save_filename}-vecnormalize.pkl")

    env.close()
