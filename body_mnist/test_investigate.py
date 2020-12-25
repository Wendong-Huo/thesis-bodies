import time
import pickle
import os
import yaml
import numpy as np
import utils
import wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize, util


def fname(train):
    if isinstance(train, list):
        fname = '-'.join(str(x) for x in train)
    else:
        fname = f"single-{train}"
    return fname


def test(test_n, seed, model_filename, vec_filename, train, test, test_as_class=0, render=False, save_file="default.yml"):

    print("Testing:")
    total_rewards = []
    distance_xs = []
    for i in range(test_n):
        print(f" Seed {seed+i}, model {model_filename} vec {vec_filename}")
        print(f" Train on {train}, test on {test}, w/ bodyinfo {test_as_class}")
        eval_env = utils.make_env(render=render, wrapper=None, robot_body=test, body_info=test_as_class)
        eval_env = DummyVecEnv([eval_env])
        eval_env = VecNormalize.load(vec_filename, eval_env)
        eval_env.norm_reward = False

        eval_env.seed(seed+i)
        model = PPO.load(model_filename)

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
        print(f"train {train}, test {test}, test_as_class {test_as_class}, step {step}, total_reward {total_reward}, distance_x {distance_x}")

        total_rewards.append(total_reward)
        distance_xs.append(distance_x)

    # avoid yaml turn float64 to numpy array
    total_rewards = [float(x) for x in total_rewards]
    distance_xs = [float(x) for x in distance_xs]

    data = {
        "title": "test",
        "train": train,
        "test": test,
        "total_reward": total_rewards,
        "distance_x": distance_xs,
    }
    with open(f"{save_file}", "w") as f:
        yaml.dump(data, f)


if __name__ == "__main__":  # noqa: C901
    args = utils.args
    utils.folder = "exp_add_diff_help"
    folder = utils.folder

    model_filename = f"checkpoints/0_48000_steps.zip"
    vec_filename = f"checkpoints/0_48000_steps.pkl"
    test(test_n=1, seed=0, model_filename=f"{folder}/{model_filename}", vec_filename=f"{folder}/{vec_filename}",
             train=0, test=0, test_as_class=0, render=True,
             save_file=f"{folder}/test-results/0.yaml")
