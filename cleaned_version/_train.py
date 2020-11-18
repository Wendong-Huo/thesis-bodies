import os
import shutil
import yaml
from collections import OrderedDict

# For custom activation fn in hyperparams
from torch import nn as nn  # noqa: F401 pytype: disable=unused-import

import torch as th
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

from arguments import get_args_train
from utils import output, delete_key

from _train_utils.load_dataset import load_dataset
from _train_utils.utils import ALGOS, linear_schedule, get_latest_run_id, create_env, SaveVecNormalizeCallback

args = get_args_train()


def _train():
    output(f"Start training with seed {args.seed}.", 1)

    output(args, 2)

    # Load hyperparameters from yaml file
    with open(f"hyperparams/default.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
    hyperparams = hyperparams_dict[args.hyperparam]

    output(f"Hyperparams: {hyperparams}", 3)

    n_envs = hyperparams.get("n_envs", 1)

    env_id, files, params, names = load_dataset(args.dataset)

    output(f"All files: {files}", 3)

    env_kwargs = {}
    if args.single:  # train on single body
        single_idx = args.single_idx

        for i in range(n_envs):
            env_kwargs[i] = {
                "xml": files[single_idx],
                "param": params[single_idx],
                "powercoeffs": [1, 1, 1],
                "render": args.watch_train and i == 0,
                "is_eval": False,
            }
        eval_env_kwargs = [{
            "xml": files[single_idx],
            "param": params[single_idx],
            "powercoeffs": [1, 1, 1],
            "render": args.watch_eval,
            "is_eval": True,
        }]
    else:  # train on a group of bodies
        # ignore hyperparams n_envs, create an env for each body
        n_envs = len(args.training_idx)
        ids = args.training_idx
        print(f"Train on bodies: {ids}")
        env_kwargs = {}
        for i in range(n_envs):
            env_kwargs[i] = {
                "xml": files[ids[i]],
                "param": params[ids[i]],
                "powercoeffs": [1, 1, 1],
                "render": args.watch_train and i == 0,
                "is_eval": False,
            }
            # Use the best body in the group to eval
            eval_env_kwargs = [{
                "xml": files[ids[0]],
                "param": params[ids[0]],
                "powercoeffs": [1, 1, 1],
                "render": args.watch_eval,
                "is_eval": True,
            }]

    output(f"Training Env: {env_kwargs}", 3)

    # Setting num threads to 1 makes things run faster on cpu
    th.set_num_threads(1)

    tensorboard_log = f"outputs/{args.exp_name}/tb/{args.exp_idx}"
    log_path = f"outputs/{args.exp_name}/logs/{args.exp_idx}"

    output(f"Training on {n_envs} environments", 2)

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

    n_timesteps = args.n_timesteps

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

    keys_to_delete = ["n_envs", "n_timesteps", "env_wrapper", "callback", "frame_stack"]
    for key in keys_to_delete:
        delete_key(hyperparams, key)

    save_path = os.path.join(log_path, f"{env_id}_{get_latest_run_id(log_path, env_id) + 1}")
    params_path = f"{save_path}/{env_id}"
    os.makedirs(params_path, exist_ok=True)

    env = create_env(n_envs, env_id, env_kwargs, seed=args.seed, normalize=True, normalize_kwargs=normalize_kwargs, eval_env=False, log_dir=log_path)

    save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=params_path)
    eval_callback = EvalCallback(
        create_env(1, env_id, eval_env_kwargs, seed=args.seed, normalize=True, normalize_kwargs=normalize_kwargs, eval_env=True),
        callback_on_new_best=save_vec_normalize,
        best_model_save_path=save_path,
        n_eval_episodes=args.eval_episodes,
        log_path=save_path,
        eval_freq=args.eval_freq,
        deterministic=True,
    )
    if args.with_bodyinfo:
        algo = "ppo_w_body"
    else:
        algo = "ppo"

    model = ALGOS[algo](env=env, tensorboard_log=tensorboard_log, seed=args.seed, verbose=True, **hyperparams)

    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    # Save hyperparams
    with open(os.path.join(params_path, "config.yml"), "w") as f:
        yaml.dump(saved_hyperparams, f)

    # save command line arguments
    with open(os.path.join(params_path, "args.yml"), "w") as f:
        ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
        yaml.dump(ordered_args, f)

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {"log_interval": args.log_interval}

    kwargs["callback"] = eval_callback
    
    output(f"n_timesteps: {n_timesteps}", 2)
    
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

    if normalize:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(os.path.join(params_path, "vecnormalize.pkl"))
        # Deprecated saving:
        # env.save_running_average(params_path)


if __name__ == "__main__":
    _train()
