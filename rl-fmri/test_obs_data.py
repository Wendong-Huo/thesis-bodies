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
g_obs_data = None

def test(seed, model_filename, vec_filename, train, test, test_as_class=0, render=False, save_file="default.yml"):
    global g_step, g_obs_data
    print("Testing:")
    total_rewards = []
    distance_xs = []
    if True:
        os.makedirs(f"{folder}/obs_data_videos", exist_ok=True)
        g_step = 0

        print(f" Seed {seed}, model {model_filename} vec {vec_filename}")
        print(f" Train on {train}, test on {test}, w/ bodyinfo {test_as_class}")
        if test_as_class>=0:
            bodyinfo = test_as_class
        else:
            if args.with_bodyinfo:
                bodyinfo = test//100
            else:
                bodyinfo = 0
        # default_wrapper = wrapper.BodyinfoWrapper
        # if args.disable_wrapper:
        #     default_wrapper = None
        default_wrapper = wrapper.WalkerWrapper

        eval_env = utils.make_env(template=utils.template(test), render=render, robot_body=test, wrapper=default_wrapper, body_info=bodyinfo)
        eval_env = DummyVecEnv([eval_env])
        if args.vec_normalize:
            eval_env = VecNormalize.load(vec_filename, eval_env)
        eval_env.norm_reward = False

        eval_env.seed(seed)
        model = PPO.load(model_filename)

        obs = eval_env.reset()
        g_obs_data = np.zeros(shape=[args.test_steps, obs.shape[1]], dtype=np.float32)

        if render:
            # eval_env.env_method("set_view")
            import common.linux
            common.linux.fullscreen()
            print("\n\nWait for a while, so I have the time to press Ctrl+F11 to enter FullScreen Mode.\n\n")
            time.sleep(2) # Wait for a while, so I have the time to press Ctrl+F11 to enter FullScreen Mode.
        distance_x = 0
        # print(obs)
        total_reward = 0
        for step in tqdm(range(args.test_steps)):
            g_obs_data[step,:] = obs[0]
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            if render:
                eval_env.envs[0].camera_adjust()
                if args.disable_saving_image:
                    time.sleep(0.01)
                else:
                    (width, height, rgbPixels, _, _) = eval_env.envs[0].pybullet.getCameraImage(1920,1080, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
                    image = rgbPixels[:,:,:3]
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{folder}/obs_data_videos/getCameraImage_b{test}_s{seed}_{step:05}.png", image)
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

        if args.save_obs_data:
            base_obs_data = None
            sorted_data = g_obs_data.copy()
            if test!=0 or seed!=0:
                # if sorted_arg_obs_data exists, use the existing one
                # because we want to compare the patterns of two experiments
                sorted_arg_obs_data = np.load(f"{folder}/sorted_arg_obs_data.npy")
                base_obs_data = np.load(f"{folder}/base_obs_data.npy")
            else:
                sorted_arg_obs_data = np.argsort(np.mean(sorted_data,axis=0))
                np.save(f"{folder}/sorted_arg_obs_data.npy", sorted_arg_obs_data)
                base_obs_data = g_obs_data.copy()
                np.save(f"{folder}/base_obs_data.npy", base_obs_data)

            # sorted_data = sorted_data[:,sorted_arg_obs_data]
            # base_obs_data = base_obs_data[:, sorted_arg_obs_data]

            for step in tqdm(range(args.test_steps)):
                plt.close()
                plt.figure(figsize=[10,4])
                if test!=0 or seed!=0:
                    x = sorted_data[step]
                    plt.bar(np.arange(len(x)), x, color=[0.1, 0.3, 0.7, 0.5])
                x = base_obs_data[step]
                plt.bar(np.arange(len(x)), x, color=[0.6, 0.6, 0.6, 0.5])
                plt.ylim(-2,2)
                plt.savefig(f"{folder}/obs_data_videos/barchart_b{test}_s{seed}_{step:05}.png")
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

    if len(args.initialize_weights_from)>0:
        model_filename = f"model-ant-{args.initialize_weights_from}.zip"
    else:
        model_filename = f"model-ant-{'-'.join(str(x) for x in train_bodies)}.zip"
    vec_filename = model_filename[:-4] + "-vecnormalize.pkl"
    os.makedirs(f"{folder}/test-results/", exist_ok=True)

    fig, axes = plt.subplots(nrows=len(test_bodies), figsize=(10,10))
    base_obs_data = None
    # Baseline
    test_body = test_bodies[0]
    test(seed=seed, model_filename=f"{folder}/{model_filename}", vec_filename=f"{folder}/{vec_filename}",
            train=train_bodies, test=test_body, test_as_class=test_as_class, render=args.render,
            save_file=f"{folder}/test-results/{model_filename[:-4]}-test-{test_body}-class-{test_as_class}.yaml")
    base_obs_data = g_obs_data.copy()
    np.save(f"{folder}/obs_data_{test_body}", base_obs_data)
    
    # Plots
    bar_colors = list(mcolors.TABLEAU_COLORS.values())
    def one_subplot(relative_obs_data, ax, title, color_idx):
        if args.save_obs_data:
            return
        ax.bar(np.arange(relative_obs_data.shape[1]), np.mean(relative_obs_data[100:900,:], axis=0), color=bar_colors[color_idx])
        ax.set_xlabel("step")
        ax.set_ylabel("activation")
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(title)

    if args.compare_seed: # compare seed
        test(seed=seed+1, model_filename=f"{folder}/{model_filename}", vec_filename=f"{folder}/{vec_filename}",
                train=train_bodies, test=test_body, test_as_class=test_as_class, render=args.render,
                save_file=f"{folder}/test-results/{model_filename[:-4]}-test-{test_body}-class-{test_as_class}.yaml")
        relative_obs_data = g_obs_data - base_obs_data
        one_subplot(relative_obs_data, axes[0], "Different Seed", 0)

    for i, test_body in enumerate(test_bodies):
        if i>0:
            test(seed=seed, model_filename=f"{folder}/{model_filename}", vec_filename=f"{folder}/{vec_filename}",
                train=train_bodies, test=test_body, test_as_class=test_as_class, render=args.render,
                save_file=f"{folder}/test-results/{model_filename[:-4]}-test-{test_body}-class-{test_as_class}.yaml")
            relative_obs_data = g_obs_data - base_obs_data
            one_subplot(relative_obs_data, axes[i], f"Difference between {test_body} and baseline", i)
        np.save(f"{folder}/obs_data_{test_body}", g_obs_data)

    if not args.save_obs_data:
        plt.tight_layout()
        plt.savefig(f"{folder}/obs_data_plot.png")
