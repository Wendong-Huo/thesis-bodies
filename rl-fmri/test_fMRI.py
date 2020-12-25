import time
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.colors as mcolors

from stable_baselines3.common.policies import ActorCriticPolicy
import yaml
import torch as th
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize, util

import common.utils as utils
import common.wrapper as wrapper
import common.plots as plots

g_step = 0
g_fMRI_data = None

def _predict_fMRI(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
    """Inject this method to ActorCriticPolicy (MlpPolicy)."""
    global g_step, g_fMRI_data
    latent_pi, _, latent_sde = self._get_latent(observation)
    # print(g_step, latent_pi)
    g_fMRI_data[g_step, :] = latent_pi
    g_step += 1
    distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
    return distribution.get_actions(deterministic=deterministic)

ActorCriticPolicy._predict = _predict_fMRI

def test(seed, model_filename, vec_filename, train, test, test_as_class=0, render=False, save_file="default.yml"):
    global g_step, g_fMRI_data
    print("Testing:")
    total_rewards = []
    distance_xs = []
    if True:
        g_step = 0
        g_fMRI_data = np.zeros(shape=[1000,256], dtype=np.float32)

        print(f" Seed {seed}, model {model_filename} vec {vec_filename}")
        print(f" Train on {train}, test on {test}, w/ bodyinfo {test_as_class}")
        if test_as_class>=0:
            bodyinfo = test_as_class
        else:
            if args.with_bodyinfo:
                bodyinfo = test//100
            else:
                bodyinfo = 0
        eval_env = utils.make_env(render=render, robot_body=test, body_info=bodyinfo)
        eval_env = DummyVecEnv([eval_env])
        eval_env = VecNormalize.load(vec_filename, eval_env)
        eval_env.norm_reward = False

        eval_env.seed(seed)
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
    folder = utils.folder

    train_bodies = [int(x) for x in args.train_bodies.split(',')]
    test_bodies = [int(x) for x in args.test_bodies.split(',')]
    test_as_class = args.test_as_class
    seed = args.seed

    model_filename = f"model-ant-{'-'.join(str(x) for x in train_bodies)}.zip"
    vec_filename = model_filename[:-4] + "-vecnormalize.pkl"
    os.makedirs(f"{folder}/test-results/", exist_ok=True)

    fig, axes = plt.subplots(nrows=len(test_bodies), figsize=(10,10))
    base_fMRI_data = None
    # Baseline
    test_body = test_bodies[0]
    test(seed=seed, model_filename=f"{folder}/{model_filename}", vec_filename=f"{folder}/{vec_filename}",
            train=train_bodies, test=test_body, test_as_class=test_as_class, render=args.render,
            save_file=f"{folder}/test-results/{model_filename[:-4]}-test-{test_body}-class-{test_as_class}.yaml")
    base_fMRI_data = g_fMRI_data.copy()
    
    # Plots
    bar_colors = list(mcolors.TABLEAU_COLORS.values())
    def one_subplot(relative_fMRI_data, ax, title, color_idx):
        ax.bar(np.arange(relative_fMRI_data.shape[1]), np.mean(relative_fMRI_data[100:900,:], axis=0), color=bar_colors[color_idx])
        ax.set_xlabel("step")
        ax.set_ylabel("activation")
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(title)

    test(seed=seed+1, model_filename=f"{folder}/{model_filename}", vec_filename=f"{folder}/{vec_filename}",
             train=train_bodies, test=test_body, test_as_class=test_as_class, render=args.render,
             save_file=f"{folder}/test-results/{model_filename[:-4]}-test-{test_body}-class-{test_as_class}.yaml")
    relative_fMRI_data = g_fMRI_data - base_fMRI_data
    one_subplot(relative_fMRI_data, axes[0], "Different Seed", 0)

    for i, test_body in enumerate(test_bodies):
        if i>0:
            test(seed=seed, model_filename=f"{folder}/{model_filename}", vec_filename=f"{folder}/{vec_filename}",
                train=train_bodies, test=test_body, test_as_class=test_as_class, render=args.render,
                save_file=f"{folder}/test-results/{model_filename[:-4]}-test-{test_body}-class-{test_as_class}.yaml")
            relative_fMRI_data = g_fMRI_data - base_fMRI_data
            one_subplot(relative_fMRI_data, axes[i], f"Difference between {test_body} and baseline", i)

    plt.tight_layout()
    plt.savefig("fMRI.png")