import time
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.colors as mcolors
from tqdm import tqdm
import cv2
import pybullet
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
        g_fMRI_data = np.zeros(shape=[args.test_steps,256], dtype=np.float32)

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
            # eval_env.env_method("set_view")
            print("\n\nWait for a while, so I have the time to press Ctrl+F11 to enter FullScreen Mode.\n\n")
            time.sleep(3) # Wait for a while, so I have the time to press Ctrl+F11 to enter FullScreen Mode.
        distance_x = 0
        # print(obs)
        total_reward = 0
        for step in tqdm(range(args.test_steps)):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            if render:
                eval_env.envs[0].camera_adjust()
                (width, height, rgbPixels, _, _) = eval_env.envs[0].env.env._p.getCameraImage(1920,1080, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
                image = rgbPixels[:,:,:3]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{folder}/fMRI_videos/getCameraImage_b{test}_s{seed}_{step:05}.png", image)
            if done:
                # it should not matter if the env reset. I guess...
                # break
                pass
            else:  # the last observation will be after reset, so skip the last
                distance_x = eval_env.envs[0].robot.body_xyz[0]
            total_reward += reward[0]
            # if render:
            #    time.sleep(0.01)

        eval_env.close()
        print(f"train {train}, test {test}, test_as_class {test_as_class}, step {step}, total_reward {total_reward}, distance_x {distance_x}")

        if args.save_fmri:
            base_fMRI_data = None
            sorted_data = g_fMRI_data.copy()
            if test!=0 or seed!=0:
                # if sorted_arg exists, use the existing one
                # because we want to compare the patterns of two experiments
                sorted_arg = np.load(f"{folder}/sorted_arg.npy")
                base_fMRI_data = np.load(f"{folder}/base_fMRI_data.npy")
            else:
                sorted_arg = np.argsort(np.mean(sorted_data,axis=0))
                np.save(f"{folder}/sorted_arg.npy", sorted_arg)
                base_fMRI_data = g_fMRI_data.copy()
                np.save(f"{folder}/base_fMRI_data.npy", base_fMRI_data)

            sorted_data = sorted_data[:,sorted_arg]
            base_fMRI_data = base_fMRI_data[:, sorted_arg]

            for step in tqdm(range(args.test_steps)):
                plt.close()
                plt.figure(figsize=[10,4])
                if test!=0 or seed!=0:
                    x = sorted_data[step]
                    plt.bar(np.arange(len(x)), x, color=[0.4, 0.7, 0.9, 0.5])
                x = base_fMRI_data[step]
                plt.bar(np.arange(len(x)), x, color=[0.3, 0.3, 0.3, 0.5])
                plt.savefig(f"{folder}/fMRI_videos/barchart_b{test}_s{seed}_{step:05}.png")
                plt.close()

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
    np.save(f"{folder}/fMRI_data_{test_body}", base_fMRI_data)
    
    # Plots
    bar_colors = list(mcolors.TABLEAU_COLORS.values())
    def one_subplot(relative_fMRI_data, ax, title, color_idx):
        ax.bar(np.arange(relative_fMRI_data.shape[1]), np.mean(relative_fMRI_data[100:900,:], axis=0), color=bar_colors[color_idx])
        ax.set_xlabel("step")
        ax.set_ylabel("activation")
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(title)

    if args.compare_seed: # compare seed
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
        np.save(f"{folder}/fMRI_data_{test_body}", g_fMRI_data)

    plt.tight_layout()
    plt.savefig(f"{folder}/fMRI.png")
