import time
import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

import common.common as common
import common.wrapper as wrapper
import common.gym_interface as gym_interface
from misalign_wrapper import MisalignedWalkerWrapper, RandomAlignedWalkerWrapper

if __name__ == "__main__":
    args = common.args
    print(args)

    assert len(args.train_bodies) == 0, "No need for body to train."

    default_wrapper = []
    if args.misalign_obs:
        default_wrapper.append(MisalignedWalkerWrapper)
    elif args.random_align_obs:
        default_wrapper.append(RandomAlignedWalkerWrapper)
    else:
        default_wrapper.append(wrapper.WalkerWrapper)
    
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
        print(obs)
        g_obs_data = np.zeros(shape=[args.test_steps, obs.shape[1]], dtype=np.float32)

        distance_x = 0
        total_reward = 0
        step = 0
        for step in tqdm(range(args.test_steps)):
            g_obs_data[step,:] = obs[0]
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

