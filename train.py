import argparse
import difflib
import importlib
import os
import time
import warnings
from collections import OrderedDict
from pprint import pprint

import gym
import numpy as np
# import seaborn
import torch as th
import yaml
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike  # noqa: F401
from stable_baselines3.common.utils import constant_fn, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

# For custom activation fn
from torch import nn as nn  # noqa: F401 pytype: disable=unused-import

# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils import ALGOS, get_latest_run_id, get_wrapper_class, linear_schedule, make_env
from utils.callbacks import SaveVecNormalizeCallback
from utils.hyperparams_opt import hyperparam_optimization
from utils.noise import LinearNormalActionNoise
from utils.utils import StoreDict, get_callback_class


# seaborn.set()

import arguments
import load_dataset
from my_utils import get_sorted_ids, get_unsorted_ids

def create_env(n_envs, eval_env=False, no_log=False):
    """
    Create the environment and wrap it if necessary
    :param n_envs: (int)
    :param eval_env: (bool) Whether is it an environment used for evaluation or not
    :param no_log: (bool) Do not log training when doing hyperparameter optim
        (issue with writing the same file)
    :return: (Union[gym.Env, VecEnv])
    """
    global hyperparams
    global env_kwargs, eval_env_kwargs
    global normalize

    if eval_env:
        kwargs = eval_env_kwargs
    else:
        kwargs = env_kwargs

    # Do not log eval env (issue with writing the same file)
    log_dir = None if eval_env or no_log else save_path

    if n_envs == 1:
        # use rank=127 so eval_env won't overlap with any training_env.
        env = DummyVecEnv(
            [make_env(env_id, 127, args.seed, wrapper_class=env_wrapper, log_dir=log_dir, env_kwargs=kwargs[0])]
        )
    else:
        # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
        # On most env, SubprocVecEnv does not help and is quite memory hungry
        env = DummyVecEnv(
            [
                make_env(env_id, i, args.seed, log_dir=log_dir, env_kwargs=kwargs[i], wrapper_class=env_wrapper)
                for i in range(n_envs)
            ]
        )

    if normalize:
        # Copy to avoid changing default values by reference
        local_normalize_kwargs = normalize_kwargs.copy()
        # Do not normalize reward for env used for evaluation
        if eval_env:
            if len(local_normalize_kwargs) > 0:
                local_normalize_kwargs["norm_reward"] = False
            else:
                local_normalize_kwargs = {"norm_reward": False}

        if args.verbose > 0:
            if len(local_normalize_kwargs) > 0:
                print(f"Normalization activated: {local_normalize_kwargs}")
            else:
                print("Normalizing input and reward")
        env = VecNormalize(env, **local_normalize_kwargs)

    return env
    
if __name__ == "__main__":  # noqa: C901
    args = arguments.get_train_args()

    # Load hyperparameters from yaml file
    with open(f"hyperparams/{args.algo}.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        if args.hyperparameters in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[args.hyperparameters]
        else:
            raise ValueError(f"Hyperparameters not found for {args.algo}-{args.hyperparameters}")

    if args.hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(args.hyperparams)
    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
    # env_kwargs = {} if args.env_kwargs is None else args.env_kwargs

    if args.verbose > 0:
        pprint(saved_hyperparams)

    n_envs = hyperparams.get("n_envs", 1)

    # Load body dataset
    dataset_name, env_id, train_files, train_params, train_names, test_files, test_params, test_names = load_dataset.load_dataset(
        args.dataset, seed=0, shuffle=False, train_proportion=1.0)

    if args.single_group < 0: # train on single body
        single_idx = args.single_idx

        env_kwargs = {}
        for i in range(n_envs):
            env_kwargs[i] = {
                "xml": train_files[single_idx],
                "param": train_params[single_idx],
                "powercoeffs": [args.powercoeff[0], args.powercoeff[1], args.powercoeff[2]],
                "render": args.watch_train and i==0,
                "is_eval": False,
            }
        eval_env_kwargs = [{
            "xml": train_files[single_idx],
            "param": train_params[single_idx],
            "powercoeffs": [args.powercoeff[0], args.powercoeff[1], args.powercoeff[2]],
            "render": args.watch_eval,
            "is_eval": True,
        }]
    else: # train on a group of bodies
        # ignore hyperparams n_envs, create an env for each body
        n_envs = 20

        ids = get_unsorted_ids()
        # ids = get_sorted_ids()
        ids = ids[args.single_group*20: args.single_group*20+20]
        print(f"Train on bodies: {ids}")
        env_kwargs = {}
        for i in range(n_envs):
            env_kwargs[i] = {
                "xml": train_files[ids[i]],
                "param": train_params[ids[i]],
                "powercoeffs": [args.powercoeff[0], args.powercoeff[1], args.powercoeff[2]],
                "render": args.watch_train and i==0,
                "is_eval": False,
            }
            # Use the best body in the group to eval
            eval_env_kwargs = [{
                "xml": train_files[ids[0]],
                "param": train_params[ids[0]],
                "powercoeffs": [args.powercoeff[0], args.powercoeff[1], args.powercoeff[2]],
                "render": args.watch_eval,
                "is_eval": True,
            }] 

    # Check seed
    assert args.seed>=0, "Please provide seed."
    set_random_seed(args.seed)
    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    tensorboard_log = None if args.tensorboard_log == "" else os.path.join(args.tensorboard_log, env_id)



    if args.verbose > 0:
        print(f"Using {n_envs} environments")

    # Create schedules
    for key in ["learning_rate", "clip_range", "clip_range_vf"]:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split("_")
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constant_fn(float(hyperparams[key]))
        else:
            raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")

    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        if args.verbose:
            print(f"Overwriting n_timesteps with n={args.n_timesteps}")
        n_timesteps = args.n_timesteps
    else:
        n_timesteps = int(hyperparams["n_timesteps"])

    normalize = False
    normalize_kwargs = {}
    if "normalize" in hyperparams.keys():
        normalize = hyperparams["normalize"]
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        if "gamma" in hyperparams:
            normalize_kwargs["gamma"] = hyperparams["gamma"]
        del hyperparams["normalize"]

    if "policy_kwargs" in hyperparams.keys():
        # Convert to python object if needed
        if isinstance(hyperparams["policy_kwargs"], str):
            hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])

    # Delete keys so the dict can be pass to the model constructor
    if "n_envs" in hyperparams.keys():
        del hyperparams["n_envs"]
    del hyperparams["n_timesteps"]

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    log_path = f"{args.log_folder}/{args.algo}/"
    save_path = os.path.join(log_path, f"{env_id}_{get_latest_run_id(log_path, env_id) + 1}")
    params_path = f"{save_path}/{env_id}"
    os.makedirs(params_path, exist_ok=True)

    callbacks = get_callback_class(hyperparams)
    if "callback" in hyperparams.keys():
        del hyperparams["callback"]

    if args.save_freq > 0:
        # Account for the number of parallel environments
        args.save_freq = max(args.save_freq // n_envs, 1)
        callbacks.append(CheckpointCallback(save_freq=args.save_freq, save_path=save_path, name_prefix="rl_model", verbose=1))

    env = create_env(n_envs)

    # Create test env if needed, do not normalize reward
    eval_callback = None
    if args.eval_freq > 0 and not args.optimize_hyperparameters:
        # Account for the number of parallel environments
        args.eval_freq = max(args.eval_freq // n_envs, 1)

        if args.verbose > 0:
            print("Creating test environment")

        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=params_path)
        eval_callback = EvalCallback(
            create_env(1, eval_env=True),
            callback_on_new_best=save_vec_normalize,
            best_model_save_path=save_path,
            n_eval_episodes=args.eval_episodes,
            log_path=save_path,
            eval_freq=args.eval_freq,
            deterministic=True,
        )

        callbacks.append(eval_callback)

    # TODO: check for hyperparameters optimization
    # TODO: check What happens with the eval env when using frame stack
    if "frame_stack" in hyperparams:
        del hyperparams["frame_stack"]

    # Stop env processes to free memory
    if args.optimize_hyperparameters and n_envs > 1:
        env.close()

    # Train an agent from scratch
    model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, seed=args.seed, verbose=args.verbose, **hyperparams)

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {"log_interval": args.log_interval}

    if len(callbacks) > 0:
        kwargs["callback"] = callbacks

    # Save hyperparams
    with open(os.path.join(params_path, "config.yml"), "w") as f:
        yaml.dump(saved_hyperparams, f)

    # save command line arguments
    with open(os.path.join(params_path, "args.yml"), "w") as f:
        ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
        yaml.dump(ordered_args, f)

    print(f"Log path: {save_path}")

    try:
        model.learn(n_timesteps, **kwargs)
    except KeyboardInterrupt:
        pass
    finally:
        # Release resources
        env.close()

    # Save trained model

    print(f"Saving to {save_path}")
    model.save(f"{save_path}/{env_id}")

    if hasattr(model, "save_replay_buffer") and args.save_replay_buffer:
        print("Saving replay buffer")
        model.save_replay_buffer(os.path.join(save_path, "replay_buffer.pkl"))

    if normalize:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(os.path.join(params_path, "vecnormalize.pkl"))
        # Deprecated saving:
        # env.save_running_average(params_path)
