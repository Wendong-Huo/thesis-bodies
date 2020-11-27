import glob
import os
import gym
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from policies.ppo_with_body_info import PPO_with_body_info
from policies.ppo_without_body_info import PPO_without_body_info

ALGOS = {
    # "a2c": A2C,
    # "ddpg": DDPG,
    # "dqn": DQN,
    # "her": HER,
    # "sac": SAC,
    # "td3": TD3,
    "ppo": PPO_without_body_info,
    "ppo_w_body": PPO_with_body_info
}

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + "/{}_[0-9]*".format(env_id)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def create_env(n_envs, env_id, kwargs, seed=0, normalize=False, normalize_kwargs=None, eval_env=False, log_dir=None):
    """
    Create the environment and wrap it if necessary
    :param n_envs: (int)
    :param eval_env: (bool) Whether is it an environment used for evaluation or not
    :param no_log: (bool) Do not log training when doing hyperparameter optim
        (issue with writing the same file)
    :return: (Union[gym.Env, VecEnv])
    """

    if n_envs == 1:
        # use rank=127 so eval_env won't overlap with any training_env.
        env = DummyVecEnv(
            [make_env(env_id, 127, seed, log_dir=log_dir, env_kwargs=kwargs)]
        )
    else:
        # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
        # On most env, SubprocVecEnv does not help and is quite memory hungry
        env = DummyVecEnv(
            [
                make_env(env_id, i, seed, log_dir=log_dir, env_kwargs=kwargs[i])
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

        env = VecNormalize(env, **local_normalize_kwargs)

    return env


def make_env(env_id, rank=0, seed=0, log_dir=None, wrapper_class=None, env_kwargs=None):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    :param wrapper_class: (Type[gym.Wrapper]) a subclass of gym.Wrapper
        to wrap the original env with
    :param env_kwargs: (Dict[str, Any]) Optional keyword argument to pass to the env constructor
    """
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    if env_kwargs is None:
        env_kwargs = {}

    def _init():
        set_random_seed(seed * 128 + rank)
        env = gym.make(env_id, **env_kwargs)

        # Wrap first with a monitor (e.g. for Atari env where reward clipping is used)
        log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
        # Monitor success rate too for the real robot
        info_keywords = ("is_success",) if "Neck" in env_id else ()
        env = Monitor(env, log_file, info_keywords=info_keywords)

        # Dict observation space is currently not supported.
        # https://github.com/hill-a/stable-baselines/issues/321
        # We allow a Gym env wrapper (a subclass of gym.Wrapper)
        if wrapper_class:
            env = wrapper_class(env)

        env.seed(seed * 128 + rank)
        return env

    return _init

class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix=None, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True
