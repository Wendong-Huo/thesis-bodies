import re
import time
import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.save_util import load_from_zip_file

from common import wrapper_custom_align, wrapper_diff, wrapper_mut, wrapper_pns
import common.common as common
import common.wrapper as wrapper
import common.gym_interface as gym_interface

from common.pns import PNSPPO, PNSMlpPolicy
from common.cnspns import CNSPNSPPO, CNSPNSPolicy

if __name__ == "__main__":
    args = common.args
    print(args)

    hyperparams = common.load_hyperparameters(conf_name="PPO")

    data, params, pytorch_variables = load_from_zip_file(args.model_filename, device="cpu")

    if args.cnspns:
        cns_parameter_means = []
        for parameter_name, module in params['policy'].items():
            _match = re.findall(r'pns_(sensor|motor)_adaptor\.nets\.([0-9]+)\.(weight|bias)', parameter_name)
            if _match:
                if _match[0][2]=='weight':
                    if _match[0][0]=="sensor":
                        print(f"Sensor channel for the policy: {module.shape[0]}")
                        args.cnspns_sensor_channel = module.shape[0]
                    else:
                        print(f"Motor channel for the policy: {module.shape[1]}")
                        args.cnspns_motor_channel = module.shape[1]
                print(f"mean of {parameter_name} is {module.numpy().mean()}")
            else:
                cns_parameter_means.append(module.numpy().mean())
        print(f"mean of all other parameters (CNS) is {np.mean(cns_parameter_means)}\n\n")

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

    if args.cnspns:
        # hard code for now. could be automatically determined.
        _w = wrapper_pns.make_same_dim_wrapper(obs_dim=28, action_dim=8)
        default_wrapper.append(_w)

    for rank_idx, test_body in enumerate(args.test_bodies):
        eval_venv = DummyVecEnv([gym_interface.make_env(rank=rank_idx, seed=common.seed, wrappers=default_wrapper, force_render=args.render,
                                                        robot_body=test_body,
                                                        dataset_folder=args.body_folder)])
        if args.vec_normalize:
            raise NotImplementedError
            # normalize_kwargs["gamma"] = hyperparams["gamma"]
            # eval_venv = VecNormalize(eval_venv, **normalize_kwargs)

        if args.stack_frames > 1:
            eval_venv = VecFrameStack(eval_venv, args.stack_frames)

        eval_venv.seed(common.seed)
        if args.pns:
            model_cls = PNSPPO
            policy_cls = PNSMlpPolicy
        elif args.cnspns:
            model_cls = CNSPNSPPO
            policy_cls = CNSPNSPolicy
        else:
            model_cls = PPO
            policy_cls = "MlpPolicy"

        model = model_cls(policy_cls, env=eval_venv, **hyperparams)
        common.load_parameters_from_path(model, args.model_filename, model_cls, args.test_bodies, default_wrapper)

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
                # eval_venv.envs[0].camera_adjust()
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

