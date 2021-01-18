import time
import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from common import wrapper_diff, wrapper_mut
import common.common as common
import common.wrapper as wrapper
import common.gym_interface as gym_interface

if __name__ == "__main__":
    args = common.args
    print(args)

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
    else:
        pass # no need for wrapper

    for test_body in args.test_bodies:
        eval_venv = DummyVecEnv([gym_interface.make_env(rank=0, seed=common.seed, wrappers=default_wrapper, render=args.render,
                                                        robot_body=test_body,
                                                        dataset_folder="../input_data/bodies")])
        if args.vec_normalize:
            raise NotImplementedError
            # normalize_kwargs["gamma"] = hyperparams["gamma"]
            # eval_venv = VecNormalize(eval_venv, **normalize_kwargs)

        if args.stack_frames > 1:
            eval_venv = VecFrameStack(eval_venv, args.stack_frames)

        eval_venv.seed(common.seed)
        model = PPO.load(args.model_filename)

        obs = eval_venv.reset()
        print(obs)
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
            if args.render:
                eval_venv.envs[0].camera_adjust()
                time.sleep(0.01)
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

