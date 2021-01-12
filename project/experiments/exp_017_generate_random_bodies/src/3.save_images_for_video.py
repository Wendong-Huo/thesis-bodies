import time
import numpy as np
from tqdm import tqdm
import cv2
import pybullet
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

import common.common as common
import common.wrapper as wrapper
import common.gym_interface as gym_interface

if __name__ == "__main__":
    args = common.args
    args.model_filename = "output_data/tmp/best_model.zip"
    args.test_bodies = [320]
    args.stack_frames = 4
    args.test_steps = 1000
    args.render = True
    print(args)

    default_wrapper = [wrapper.WalkerWrapper]

    assert len(args.train_bodies) == 0, "No need for body to train."
    if args.with_bodyinfo:
        default_wrapper += [wrapper.BodyinfoWrapper]

    for test_body in args.test_bodies:
        eval_venv = DummyVecEnv([gym_interface.make_env(rank=0, seed=common.seed, wrappers=default_wrapper, render=args.render,
                                                        robot_body=test_body)])
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

        if True:
            import common.linux
            common.linux.fullscreen()
            print("\n\nWait for a while, so I have the time to press Ctrl+F11 to enter FullScreen Mode.\n\n")
            time.sleep(2) # Wait for a while, so I have the time to press Ctrl+F11 to enter FullScreen Mode.

        distance_x = 0
        total_reward = 0
        step = 0
        for step in tqdm(range(args.test_steps)):
            g_obs_data[step,:] = obs[0]
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_venv.step(action)
            if args.render:
                eval_venv.envs[0].camera_adjust()

                (width, height, rgbPixels, _, _) = eval_venv.envs[0].pybullet.getCameraImage(1920,1080, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
                image = rgbPixels[:,:,:3]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"output_data/saved_images/getCameraImage_{step:05}.png", image)

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

        for step in tqdm(range(args.test_steps)):
            plt.close()
            plt.figure(figsize=[10,4])
            # if test!=0 or seed!=0:
            #     x = g_obs_data[step]
            #     plt.bar(np.arange(len(x)), x, color=[0.1, 0.3, 0.7, 0.5])
            x = g_obs_data[step]
            plt.bar(np.arange(len(x)), x, color=[0.6, 0.6, 0.6, 0.5])
            plt.ylim(-2,2)
            plt.savefig(f"output_data/saved_images/barchart_{step:05}.png")
            plt.close()