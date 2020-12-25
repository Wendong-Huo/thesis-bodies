import os, time
from stable_baselines3.common import callbacks
import yaml
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

from callbacks.eval import EvalCallback_with_prefix
import utils

def test(seed, model, train, test, normalize_kwargs, body_info=0, render=False):
    print("Testing:")
    print(f" Train on {train}, test on {test}, w/ bodyinfo {body_info}")
    eval_env = DummyVecEnv([utils.make_env(rank=0, seed=utils.seed+1, render=False, robot_body=test, body_info=0)])
    eval_env = VecNormalize(eval_env, norm_reward=False, **normalize_kwargs)
    eval_env.seed(seed)

    obs = eval_env.reset()
    if render:
        eval_env.env_method("set_view")
    distance_x = 0
    # print(obs)
    total_reward = 0
    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done:
            break
        else:  # the last observation will be after reset, so skip the last
            distance_x = eval_env.envs[0].robot.body_xyz[0]
        total_reward += reward[0]
        if render:
            time.sleep(0.01)

    eval_env.close()
    print(f"train {train}, test {test}, body_info {body_info}, step {step}, total_reward {total_reward}, distance_x {distance_x}")
    return total_reward, distance_x

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
    train_num_envs = 16 if not debug else 2
    total_timesteps = 1e6 if not debug else 1

    training_bodies = [int(x) for x in args.train_bodies.split(",")]
    str_ids = "-".join(str(x) for x in training_bodies)
    test_bodies = [int(x) for x in args.test_bodies.split(",")]
    
    assert len(training_bodies)==1 and len(test_bodies)==1
    env = DummyVecEnv([utils.make_env(rank=i, seed=utils.seed, render=args.render, robot_body=training_bodies[i%len(training_bodies)], body_info=0) for i in range(train_num_envs)])
    save_filename = f"model-ant-{str_ids}"

    print(save_filename)

    env = VecNormalize(env, **normalize_kwargs)

    keys_remove =["normalize", "n_envs", "n_timesteps", "policy"]
    for key in keys_remove:
        del hyperparams[key]

    all_callbacks = []
    for test_body in test_bodies:
        eval_env = DummyVecEnv([utils.make_env(rank=0, seed=utils.seed+1, render=False, robot_body=test_body, body_info=0)])
        eval_env = VecNormalize(eval_env, norm_reward=False, **normalize_kwargs)
        eval_callback = EvalCallback_with_prefix(
            eval_env=eval_env,
            prefix=f"{test_body}",
            n_eval_episodes=3,
            eval_freq=1e3, # will implicitly multiplied by 16 (train_num_envs)
            deterministic=True,
        )
        all_callbacks.append(eval_callback)

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"{folder}/tb/{save_filename}", seed=utils.seed, **hyperparams)

    model.learn(total_timesteps=total_timesteps, callback=all_callbacks)

    total_reward, distance_x = test(seed=8, model=model, train=training_bodies[0], test=test_bodies[0], normalize_kwargs=normalize_kwargs)

    if total_reward>1500 and distance_x>20: # threshold for the definition of workable
        with open(f"{folder}/works-{save_filename}", "w") as f:
            print(f"total_reward = {total_reward}, distance_x = {distance_x}", file=f)
    model.save(f"{folder}/{save_filename}")
    # Important: save the running average, for testing the agent we need that normalization
    model.get_vec_normalize_env().save(f"{folder}/{save_filename}-vecnormalize.pkl")
    env.close()
