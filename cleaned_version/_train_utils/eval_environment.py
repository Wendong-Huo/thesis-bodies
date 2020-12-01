import os
import yaml
import re

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
from .utils import make_env
from .utils import linear_schedule
from utils import output, delete_key

# A hack. I record two functions into config.yml.
import _train_utils.utils
import stable_baselines3.common.utils
def null(**kwargs):
    pass
_train_utils.utils.func = null
stable_baselines3.common.utils.func = null
# A hack. I record two functions into config.yml.


def get_saved_hyperparams(stats_path, norm_reward=False, test_mode=False):
    """
    :param stats_path: (str)
    :param norm_reward: (bool)
    :param test_mode: (bool)
    :return: (dict, str)
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
        output(f"Not a valid path {stats_path}.",-1)
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, "config.yml"), "r") as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        else:
            obs_rms_path = os.path.join(stats_path, "obs_rms.pkl")
            hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                if test_mode:
                    normalize_kwargs["norm_reward"] = norm_reward
            else:
                normalize_kwargs = {"norm_obs": hyperparams["normalize"], "norm_reward": norm_reward}
            hyperparams["normalize_kwargs"] = normalize_kwargs
    return hyperparams, stats_path

def create_test_env(
    env_id, n_envs=1, stats_path=None, seed=0, log_dir="", should_render=True, hyperparams=None, env_kwargs=None
):
    """
    Create environment for testing a trained agent

    :param env_id: (str)
    :param n_envs: (int) number of processes
    :param stats_path: (str) path to folder containing saved running averaged
    :param seed: (int) Seed for random number generator
    :param log_dir: (str) Where to log rewards
    :param should_render: (bool) For Pybullet env, display the GUI
    :param hyperparams: (dict) Additional hyperparams (ex: n_stack)
    :param env_kwargs: (Dict[str, Any]) Optional keyword argument to pass to the env constructor
    :return: (gym.Env)
    """
    # HACK to save logs
    # if log_dir is not None:
    #     os.environ["OPENAI_LOG_FORMAT"] = 'csv'
    #     os.environ["OPENAI_LOGDIR"] = os.path.abspath(log_dir)
    #     os.makedirs(log_dir, exist_ok=True)
    #     logger.configure()

    # Clean hyperparams, so the dict can be pass to the model constructor
    if True:
        keys_to_delete = ["n_envs", "n_timesteps", "env_wrapper", "callback", "frame_stack"]
        for key in keys_to_delete:
            delete_key(hyperparams, key)

    if n_envs > 1:
        # start_method = 'spawn' for thread safe
        env = SubprocVecEnv(
            [make_env(env_id, i, seed, log_dir, env_kwargs=env_kwargs) for i in range(n_envs)]
        )
    # Pybullet envs does not follow gym.render() interface
    elif "Bullet" in env_id or "Walker2D" in env_id:
        # HACK: force SubprocVecEnv for Bullet env
        env = DummyVecEnv([make_env(env_id, 127, seed, log_dir, env_kwargs=env_kwargs)])
    else:
        env = DummyVecEnv([make_env(env_id, 127, seed, log_dir, env_kwargs=env_kwargs)])

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams["normalize"]:
            # print("Loading running average")
            # print("with params: {}".format(hyperparams["normalize_kwargs"]))
            path_ = os.path.join(stats_path, "vecnormalize.pkl")
            if os.path.exists(path_):
                env = VecNormalize.load(path_, env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)
    return env