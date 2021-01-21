

import pyrobotdesign as rd
import pyrobotdesign_env
import time,os
import numpy as np
from tqdm import tqdm


from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from common import wrapper_custom_align, wrapper_diff, wrapper_mut

import common.common as common
import common.wrapper as wrapper
import common.gym_interface as gym_interface
import common.callbacks as callbacks
from common.activation_fn import MyThreshold
from common.pns import PNSPPO, PNSMlpPolicy

if __name__ == "__main__":

    args = common.args
    print(args)

    # args.vec_normalize = True # Robo need normalization.

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

    if args.with_bodyinfo:
        default_wrapper.append(wrapper.BodyinfoWrapper)
    assert len(args.robo_bodies) > 0, "No body to test."

    print("Making eval environments...")
    for rank_idx, test_body in enumerate(args.robo_bodies):
        body_info = 0
        eval_venv = DummyVecEnv([gym_interface.make_pyrobotdesign_env(rank=rank_idx, seed=common.seed+1, wrappers=default_wrapper, render=args.render,
                                                        robo_body=test_body, dataset_folder=args.dataset_folder)])
        if args.vec_normalize:
            vorm_filename = f"{args.model_filename[:-4]}.vnorm.pkl"
            eval_venv = VecNormalize.load(vorm_filename, eval_venv)
        if args.stack_frames > 1:
            eval_venv = VecFrameStack(eval_venv, args.stack_frames)

        eval_venv.seed(common.seed)
        if args.pns:
            model_cls = PNSPPO
            policy_cls = PNSMlpPolicy
        else:
            model_cls = PPO
            policy_cls = "MlpPolicy"

        model = model_cls.load(args.model_filename, env=eval_venv)

        obs = eval_venv.reset()
        print(obs.shape)
        g_obs_data = np.zeros(shape=[args.test_steps, obs.shape[1]], dtype=np.float32)

        distance_x = 0
        total_reward = 0
        step = 0
        for step in tqdm(range(args.test_steps)):
            g_obs_data[step,:] = obs[0]
            # for i in obs[0]:
            #     print(f"{i:.02f}", end=", ")
            # print("")
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_venv.step(action)
            print(action.mean())
            if args.render:
                # eval_venv.envs[0].camera_adjust()
                time.sleep(0.01)
            if done:
                # it should not matter if the env reset. I guess...
                # break
                pass
            else:  # the last observation will be after reset, so skip the last
                distance_x = -1
            total_reward += reward[0]

        eval_venv.close()
        print(f"model filename: {args.model_filename}")
        print(f"test on {test_body}")
        print(f"Results: last step {step}, total_reward {total_reward}, distance_x {distance_x}")
        print("\n"*4)
