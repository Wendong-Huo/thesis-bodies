from pathlib import Path

import yaml
import torch.nn as nn
import gym
from stable_baselines3.common.utils import set_random_seed

body_names = []
for i in range(10):
    body_names.append(f"walker2d_{i}")

def xml_filename(i):
    assert 0 <= i < len(body_names)
    return f"{Path.cwd()}/bodies/{body_names[i]}.xml"


def make_env(env_id, rank, xml, seed=0, render=True, is_eval=False, param=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render=render, debug=False, xml=xml, is_eval=is_eval, param=param)
        env.isRender = render
        env.seed(seed + rank)
        env.reset()
        return env
    set_random_seed(seed)
    return _init

def load_hyperparams(RL_method_str):
    with open(f"hyperparams/tuned.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
    hyperparams = hyperparams_dict[RL_method_str]

    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    if "policy_kwargs" in hyperparams.keys():
        # Convert to python object if needed
        if isinstance(hyperparams["policy_kwargs"], str):
            hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])

    # TODO: do I need to parse and use noise?
    if "noise_type" in hyperparams.keys():
        del hyperparams["noise_type"]
    if "noise_std" in hyperparams.keys():
        del hyperparams["noise_std"]

    return hyperparams
