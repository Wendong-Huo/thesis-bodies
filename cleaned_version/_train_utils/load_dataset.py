import os
import pickle
from pathlib import Path

import yaml
import numpy as np
import gym
from gym import error

from utils import output, read_yaml

def load_dataset(path):
    config = read_yaml(f"{path}/config.yaml")

    dataset_name = config["dataset_name"]
    # params
    params_files = config["bodies"]["params"]
    params = []
    for p in params_files:
        with open(f"{Path.cwd()}/{path}/{p}", "r") as f:
            if p[-5:] == ".yaml":
                _dic = yaml.safe_load(f)
                param = list(_dic.values())
                if np.max(param) > 3.0:
                    output("Warning: large parameters. Please consider normalize the value.", 1)
                params.append(_dic)
            else:
                params.append(pickle.load(f))

    # body files
    files = config["bodies"]["files"]
    for f in files:
        if not os.path.exists(f"{Path.cwd()}/{path}/{f}"):
            print(f"Dataset is not valid. {Path.cwd()}/{path}/{f} not found.")

    files = [ f"{Path.cwd()}/{path}/{f}" for f in files]
    # gym env
    gym_env_filename = f"{path}/{config['gym_env']['filename']}"
    gym_env_class = config["gym_env"]["class"]
    gym_env_id = config["gym_env"]["env_id"]
    try:
        gym.spec(gym_env_id)
        # print("Already registered. Skip.")
    except error.UnregisteredEnv:
        register_env(gym_env_id, gym_env_filename, gym_env_class)

    # return dataset_name, gym_env_id, train_files, train_params, train_names, test_files, test_params, test_names
    return gym_env_id, files, params, body_names(files)

def register_env(gym_env_id, gym_env_filename, gym_env_class):
    if not os.path.exists(gym_env_filename):
        print(f"Dataset env file not found. {gym_env_filename}")
    
    assert gym_env_filename[-3:] == ".py"
    gym_env_filename = gym_env_filename[:-3] # remove ".py"
    gym_env_filename = gym_env_filename.replace("/",".") # turn file path into python class path
    gym.register(id=gym_env_id, entry_point=f"{gym_env_filename}:{gym_env_class}", kwargs={'name': 0, 'render': False, 'xml': "", 'param': []})

def body_names(filenames):
    names = []
    for filename in filenames:
        assert filename[-4:]==".xml"
        filename = filename[:-4]
        tmp = filename.split("/")
        filename = tmp[-1]
        names.append(filename)
    return names