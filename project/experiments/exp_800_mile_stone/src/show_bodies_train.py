import cv2
import numpy as np
import time
import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.save_util import load_from_zip_file

from common import wrapper_custom_align, wrapper_diff, wrapper_mut, wrapper_pns

import common.common as common
from common.utils import linux_fullscreen, load_parameters_from_path
import common.wrapper as wrapper
import common.gym_interface as gym_interface
import common.callbacks as callbacks
from common.activation_fn import MyThreshold
from common.pns import PNSPPO, PNSMlpPolicy
from common.cnspns import CNSPNSPPO, CNSPNSPolicy

if __name__ == "__main__":

    args = common.args
    print(args)

    args.test_bodies = args.train_bodies # omit test_bodies in command from now on.
    args.initialize_weights_from = args.model_filename

    # SAC.learn need this. If use SubprocVecEnv instead of DummyVecEnv, you need to seed in each subprocess.
    set_random_seed(common.seed)

    saved_model_filename = common.build_model_filename(args)

    hyperparams = common.load_hyperparameters(conf_name=args.rl_hyperparameter)
    print(hyperparams)

    # Make every env has the same obs space and action space
    default_wrapper = []
    # if padding zero:
    #   default_wrapper.append(wrapper.WalkerWrapper)

    if args.topology_wrapper == "same":
        body_type = 0
        for body in args.train_bodies + args.test_bodies:
            if body_type == 0:
                body_type = body//100
            else:
                assert body_type == body//100, "Training on different body types."
        if args.realign_method != "":
            default_wrapper.append(wrapper.ReAlignedWrapper)
    elif args.topology_wrapper == "diff":
        default_wrapper.append(wrapper_diff.get_wrapper_class())
    elif args.topology_wrapper == "MutantWrapper":
        default_wrapper.append(wrapper_mut.MutantWrapper)
    elif args.topology_wrapper == "CustomAlignWrapper":
        default_wrapper.append(wrapper_custom_align.CustomAlignWrapper)
    else:
        pass  # no need for wrapper

    # if args.cnspns:
    # hard code for now. could be automatically determined.
    _w = wrapper_pns.make_same_dim_wrapper(obs_dim=28, action_dim=8)
    default_wrapper.append(_w)

    assert len(args.train_bodies) > 0, "No body to train."
    if args.with_bodyinfo:
        default_wrapper.append(wrapper.BodyinfoWrapper)

    photo_idx = 2

    print("Making train environments...")
    venv = DummyVecEnv([gym_interface.make_env(rank=i, seed=common.seed, wrappers=default_wrapper, render=args.render,
                                               robot_body=args.train_bodies[i % len(args.train_bodies)],
                                               dataset_folder=args.body_folder,
                                               render_index=photo_idx,
                                               ) for i in range(args.num_venvs)])
    venv.reset()
    p = venv.envs[photo_idx].env.env.env._p
    # for i in range(8):
    #     for j in range(100):
    #         action = np.zeros(shape=[4,8])
    #         action[0,i] = 1
    #         venv.step(action)
    #         time.sleep(0.01)

    for i in range(28):
        action = np.zeros(shape=[4,8])
        for n in [0,1]:
            action[0,n] = -1

        for n in [1,3,7]:
            action[2,n] = -0.2
        for n in [0,4,5]:
            action[2,n] = 0.2
        
        action[3,:] = -0.1
        venv.step(action)
    linux_fullscreen()
    time.sleep(1)
    if True:
        (width, height, rgbPixels, _, _) = p.getCameraImage(1920,1080, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        image = rgbPixels[:,:,:3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"output_data/saved_images/{gym_interface.template(args.train_bodies[photo_idx])}_{args.custom_alignment}.png", image)
        # break  