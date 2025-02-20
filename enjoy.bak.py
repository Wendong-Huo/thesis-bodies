import argparse
import importlib
import os
from time import sleep

import gym
import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.utils import StoreDict
from utils.wrappers import TimeFeatureWrapper

import load_dataset

def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="Walker2DBulletEnv-v0")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )

    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions (for DDPG/DQN/SAC)")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    # === #
    parser.add_argument("--load-checkpoint", type=str, help="pass the path of zip file corresponding to it")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--dataset", type=str, default="dataset/walker2d_v6")
    parser.add_argument("--body-id", type=int, default=0)
    args = parser.parse_args()

    dataset_name, env_id, train_files, train_params, train_names, test_files, test_params, test_names = load_dataset.load_dataset(
        args.dataset, seed=0, shuffle=False, train_proportion=1)

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    # env_id = args.env
    algo = args.algo
    log_path = args.folder

    # if args.exp_id == 0:
    #     args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
    #     print(f"Loading latest experiment, id={args.exp_id}")

    # # Sanity checks
    # if args.exp_id > 0:
    #     log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    # else:
    #     log_path = os.path.join(folder, algo)

    # assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    # found = False
    # for ext in ["zip"]:
    #     model_path = os.path.join(log_path, f"{env_id}.{ext}")
    #     found = os.path.isfile(model_path)
    #     if found:
    #         break

    # if args.load_best:
    #     model_path = os.path.join(log_path, "best_model.zip")
    #     found = os.path.isfile(model_path)

    # if args.load_checkpoint is not None:
    #     model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
    #     found = os.path.isfile(model_path)

    # if not found:
    #     raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    model_path = args.load_checkpoint

    if algo in ["dqn", "ddpg", "sac", "td3", "tqc"]:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = "NoFrameskip" in env_id

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    # env_kwargs = {}
    # args_path = os.path.join(log_path, env_id, "args.yml")
    # if os.path.isfile(args_path):
    #     with open(args_path, "r") as f:
    #         loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
    #         if loaded_args["env_kwargs"] is not None:
    #             env_kwargs = loaded_args["env_kwargs"]
    # # overwrite with command line arguments
    # if args.env_kwargs is not None:
    #     env_kwargs.update(args.env_kwargs)

    args.watch_eval = True

    env_kwargs = {
        "xml": train_files[args.body_id],
        "param": train_params[args.body_id],
        "render": args.watch_eval,
    }
    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in ["dqn", "ddpg", "sac", "her", "td3", "tqc"]:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    model = ALGOS[algo].load(model_path, env=env, **kwargs)

    obs = env.reset()

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in ["dqn", "ddpg", "sac", "her", "td3", "tqc"] and not args.stochastic

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    for _ in range(args.n_timesteps):
        action, state = model.predict(obs, state=state, deterministic=deterministic)
        # Random Agent
        # action = [env.action_space.sample()]
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        sleep(0.01)
        if not args.no_render:
            env.render("human")

        episode_reward += reward[0]
        ep_len += 1

        if args.n_envs == 1:
            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            if is_atari and infos is not None and args.verbose >= 1:
                episode_infos = infos[0].get("episode")
                if episode_infos is not None:
                    print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                    print("Atari Episode Length", episode_infos["l"])

            if done and not is_atari and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print(f"Episode Reward: {episode_reward:.2f}")
                print("Episode Length", ep_len)
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)
                episode_reward = 0.0
                ep_len = 0
                state = None

            # Reset also when the goal is achieved when using HER
            if done and infos[0].get("is_success") is not None:
                if args.verbose > 1:
                    print("Success?", infos[0].get("is_success", False))
                # Alternatively, you can add a check to wait for the end of the episode
                if done:
                    obs = env.reset()
                if infos[0].get("is_success") is not None:
                    successes.append(infos[0].get("is_success", False))
                    episode_reward, ep_len = 0.0, 0

    if args.verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))

    if args.verbose > 0 and len(episode_lengths) > 0:
        print("Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and "Bullet" not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecEnvWrapper):
                env = env.venv
            if isinstance(env, DummyVecEnv):
                env.envs[0].env.close()
            else:
                env.close()
        else:
            # SubprocVecEnv
            env.close()


if __name__ == "__main__":
    main()
