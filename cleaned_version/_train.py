import os
import shutil
import yaml
from collections import OrderedDict

import numpy as np
# For custom activation fn in hyperparams
from torch import nn as nn  # noqa: F401 pytype: disable=unused-import

import torch as th
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

from arguments import get_args_train
from utils import output, delete_key
from callbacks.callbacks import DumpWeightsCallback

from _train_utils.load_dataset import load_dataset
from _train_utils.utils import ALGOS, linear_schedule, get_latest_run_id, create_env, SaveVecNormalizeCallback

args = get_args_train()


def _train():
    output(f"Start training with seed {args.seed}.", 1)

    output(args, 2)

    # Setting num threads to 1 makes things run faster on cpu, and we only use one cpu for one training.
    th.set_num_threads(1)

    # Load hyperparameters from yaml file
    with open(f"hyperparams/default.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
    hyperparams = hyperparams_dict[args.hyperparam]

    output(f"Hyperparams: {hyperparams}", 3)

    n_envs = hyperparams.get("n_envs", 1)

    env_id, files, params, names = load_dataset(args.dataset)

    output(f"All files: {files}", 3)

    # Creating Environments, both training and evaluating.
    if True:
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
            ids = np.fromstring(args.body_ids, dtype=int, sep=',')
            n_envs = len(ids)
            output(f"Train on bodies: {ids}", 2)
            env_kwargs = {}
            for i in range(n_envs):
                env_kwargs[i] = {
                    "xml": files[ids[i]],
                    "param": params[ids[i]],
                    "name": ids[i],
                    "powercoeffs": [1, 1, 1],
                    "render": args.watch_train and i == 0,
                    "is_eval": False,
                }
            eval_ids = np.fromstring(args.eval_ids, dtype=int, sep=',')
            eval_n_envs = len(eval_ids)
            output(f"Evaluate on bodies: {eval_ids}", 2)
            eval_env_kwargs = {}
            for i in range(eval_n_envs):
                # Use the best body in the group to eval
                eval_env_kwargs[i] = {
                    "xml": files[eval_ids[i]],
                    "param": params[eval_ids[i]],
                    "name": eval_ids[i],
                    "powercoeffs": [1, 1, 1],
                    "render": args.watch_eval and i == 0,
                    "is_eval": True,
                }

        output(f"Training Env: {env_kwargs}", 3)

    # Setting Pathes
    if True:
        mode = "single" if args.single else "multi"
        if args.with_bodyinfo:
            mode += "_body"
        tensorboard_log = f"outputs/{args.exp_name}/tb/{mode}/i{args.exp_idx}_s{args.seed}"
        log_path = f"outputs/{args.exp_name}/logs/{mode}/i{args.exp_idx}_s{args.seed}"
        save_path = os.path.join(log_path, f"{env_id}_{get_latest_run_id(log_path, env_id) + 1}")
        params_path = f"{save_path}/{env_id}"
        os.makedirs(params_path, exist_ok=True)

    output(f"Training on {n_envs} environments", 1)

    # Adjusting hyperparameters
    if True:
        if not args.single:
            # because we are training on many bodies, we need larger buffer size to save the replay experience, to avoid the training go diverge.
            # A nice explanation: https://stats.stackexchange.com/questions/265964/why-is-deep-reinforcement-learning-unstable
            hyperparams["n_steps"] = hyperparams["n_steps"] * n_envs

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

    # Clean hyperparams, so the dict can be pass to the model constructor
    if True:
        keys_to_delete = ["n_envs", "n_timesteps", "env_wrapper", "callback", "frame_stack"]
        for key in keys_to_delete:
            delete_key(hyperparams, key)

    # Evaluation Environments
    if True:
        # Eval right before dumping the log:
        eval_freq = hyperparams["n_steps"] * args.log_interval

        all_callbacks = []
        dump_callback = DumpWeightsCallback()
        all_callbacks.append(dump_callback)
        for i, _kwargs in enumerate(eval_env_kwargs):
            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=params_path)
            eval_callback = EvalCallback(
                create_env(1, env_id, eval_env_kwargs[i], seed=args.seed, normalize=True, normalize_kwargs=normalize_kwargs, eval_env=True),
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=save_path,
                n_eval_episodes=args.eval_episodes,
                log_path=save_path,
                eval_freq=eval_freq,
                deterministic=True,
            )
            all_callbacks.append(eval_callback)


    # Start training
    if True:
        env = create_env(n_envs, env_id, env_kwargs, seed=args.seed, normalize=True, normalize_kwargs=normalize_kwargs, eval_env=False, log_dir=log_path)
        algo = "ppo_w_body" if args.with_bodyinfo else "ppo"
        model = ALGOS[algo](env=env, tensorboard_log=tensorboard_log, seed=args.seed, verbose=True, **hyperparams)

    # Save params and arguments
    if True:
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        # Save hyperparams
        # TODO: don't save some items and need to save same items.
        with open(os.path.join(params_path, "config.yml"), "w") as f:
            yaml.dump(saved_hyperparams, f)

        # save command line arguments
        with open(os.path.join(params_path, "args.yml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
            yaml.dump(ordered_args, f)

    # Make a joke! To see how much the weights change during training.
    if True:
        d = model.policy.mlp_extractor.policy_net._modules["0"].weight.data
        import imageio
        _weights = imageio.imread("weights/39x256.png")
        _weights = (_weights[:,:,0] / 256.0 * 0.32 - 0.16).astype(np.float32)
        if args.single:
            _weights = _weights[:,:22]
        print(model.policy.mlp_extractor.policy_net._modules["0"].weight.data.shape)
        model.policy.mlp_extractor.policy_net._modules["0"].weight.data = th.from_numpy(_weights)

    # Start training
    if True:
        kwargs = {}
        if args.log_interval > -1:
            kwargs = {"log_interval": args.log_interval}

        kwargs["callback"] = all_callbacks

        output(f"n_timesteps: {n_timesteps}", 2)

        try:
            model.learn(n_timesteps, **kwargs)
        except KeyboardInterrupt:
            pass
        finally:
            # Release resources
            env.close()

    # Save trained model
    if True:
        print(f"Saving to {save_path}")
        model.save(f"{save_path}/{env_id}")

        if normalize:
            # Important: save the running average, for testing the agent we need that normalization
            model.get_vec_normalize_env().save(os.path.join(params_path, "vecnormalize.pkl"))
            # Deprecated saving:
            # env.save_running_average(params_path)


if __name__ == "__main__":
    _train()
