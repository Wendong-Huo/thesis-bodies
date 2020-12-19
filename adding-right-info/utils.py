import os
import yaml
import torch.nn as nn
import gym
import wrapper
from stable_baselines3.common.callbacks import BaseCallback
import arguments

args = arguments.get_args()
if args.exp != "":
    folder = args.exp
else:
    folder = "exp_v3"
seed = 1

def make_env(rank=0, seed=0, render=True, wrapper=wrapper.BodyinfoWrapper, robot_body=-1, body_info=-1):
    def _init():
        try:
            gym.spec(f'AntBulletEnv-v{robot_body}')
        except:
            gym.envs.registration.register(id=f'AntBulletEnv-v{robot_body}',
                entry_point=f'{folder}.envs.ant:AntBulletEnv',
                max_episode_steps=1000,
                reward_threshold=2500.0, 
                kwargs={"xml":f"{os.getcwd()}/{folder}/envs/{robot_body}.xml"})
        _render = False
        if render:
            _render = rank in [0]
        env = gym.make(f'AntBulletEnv-v{robot_body}', render=_render)
        if wrapper is not None:
            if body_info<0:
                _body_info = robot_body
            else:
                _body_info = body_info
            env = wrapper(env, _body_info)

        if seed is not None:
            env.seed(seed*100 + rank)
            env.action_space.seed(seed*100 + rank)
        return env

    return _init

def load_hyperparameters():
    with open("hyperparameters.yml", "r") as f:
        hp = yaml.load(f, Loader=yaml.SafeLoader)
    hyperparams = hp["AntBulletEnv-v0"]
    hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
    return hyperparams

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
