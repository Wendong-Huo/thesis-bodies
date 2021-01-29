import os, pathlib
import time
import numpy as np
from tqdm import tqdm
import cv2
import pybullet
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from common import wrapper_custom_align, wrapper_diff, wrapper_mut

import common.common as common
import common.wrapper as wrapper
import common.gym_interface as gym_interface
from common import linux

if __name__ == "__main__":
    args = common.args
    # args.model_filename = "output_data/tmp/best_model.zip"
    # args.test_bodies = [320]
    # args.stack_frames = 4
    # args.test_steps = 10
    # args.render = True
    print(args)
    model_name = pathlib.Path(args.model_filename).stem
    os.makedirs(f"output_data/saved_images/{model_name}", exist_ok=False)

    assert len(args.train_bodies) == 0, "No need for body to train."

    # Make every env has the same obs space and action space
    default_wrapper = []
    # if padding zero:
    #   default_wrapper.append(wrapper.WalkerWrapper)
    
    if args.topology_wrapper == "same":
        body_type = 0
        for body in args.train_bodies + args.test_bodies:
            if body_type==0:
                body_type = body//100
            else:
                assert body_type == body//100, "Training on different body types."
        if args.realign_method!="":
            default_wrapper.append(wrapper.ReAlignedWrapper)
    elif args.topology_wrapper == "diff":
        default_wrapper.append(wrapper_diff.get_wrapper_class())
    elif args.topology_wrapper == "MutantWrapper":
        default_wrapper.append(wrapper_mut.MutantWrapper)
    elif args.topology_wrapper == "CustomAlignWrapper":
        default_wrapper.append(wrapper_custom_align.CustomAlignWrapper)
    else:
        pass # no need for wrapper


    for rank_idx, test_body in enumerate(args.test_bodies):
        eval_venv = DummyVecEnv([gym_interface.make_env(rank=rank_idx, seed=common.seed, wrappers=default_wrapper, render=args.render, force_render=args.render,
                                                        robot_body=test_body,
                                                        dataset_folder=args.body_folder)])
        if args.vec_normalize:
            raise NotImplementedError
            # normalize_kwargs["gamma"] = hyperparams["gamma"]
            # eval_venv = VecNormalize(eval_venv, **normalize_kwargs)

        if args.stack_frames > 1:
            eval_venv = VecFrameStack(eval_venv, args.stack_frames)

        eval_venv.seed(common.seed)
        model = PPO.load(args.model_filename)

        obs = eval_venv.reset()
        g_obs_data = np.zeros(shape=[args.test_steps, obs.shape[1]], dtype=np.float32)

        time.sleep(1)
        linux.fullscreen()
        time.sleep(1)

        distance_x = 0
        total_reward = 0
        step = 0
        for step in tqdm(range(args.test_steps)):
            g_obs_data[step,:] = obs[0]
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_venv.step(action)
            if args.render:
                eval_venv.envs[0].camera_adjust()
                if common.args.one_snapshot_at==-1 or common.args.one_snapshot_at==step:
                    (width, height, rgbPixels, _, _) = eval_venv.envs[0].pybullet.getCameraImage(1920,1080, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
                    image = rgbPixels[:,:,:3]
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"output_data/saved_images/{model_name}/test_{test_body}_{step:05}.png", image)

            if done:
                # it should not matter if the env reset. I guess...
                break
                # pass
            else:  # the last observation will be after reset, so skip the last
                distance_x = eval_venv.envs[0].robot.body_xyz[0]
            total_reward += reward[0]

        eval_venv.close()
        print(f"model filename: {args.model_filename}")
        print(f"test on {test_body}")
        print(f"Results: last step {step}, total_reward {total_reward}, distance_x {distance_x}")
        print("\n"*4)

        record_observations = False
        if record_observations:
            for step in tqdm(range(args.test_steps)):
                plt.close()
                plt.figure(figsize=[10,4])
                # if test!=0 or seed!=0:
                #     x = g_obs_data[step]
                #     plt.bar(np.arange(len(x)), x, color=[0.1, 0.3, 0.7, 0.5])
                x = g_obs_data[step]
                plt.bar(np.arange(len(x)), x, color=[0.6, 0.6, 0.6, 0.5])
                plt.ylim(-2,2)
                plt.savefig(f"output_data/saved_images/barchart_{test_body}_{step:05}.png")
                plt.close()
