import os
import pickle
from pathlib import Path

import yaml
import numpy as np
import gym

def load_walker2d(seed=0, shuffle=True):
    return load_dataset("dataset/walker_x", seed=seed, shuffle=shuffle)

def load_dataset(path, train_proportion=0.8, seed=0, shuffle=True):
    with open(f"{path}/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset_name = config["dataset_name"]
    # params
    params_files = config["bodies"]["params"]
    params = []
    for p in params_files:
        with open(f"{Path.cwd()}/{path}/{p}", "r") as f:
            if p[-5:] == ".yaml":
                _dic = yaml.safe_load(f)
                param = list(_dic.values())
                if np.max(param) > 2.0:
                    print("Warning: large parameters. Please consider normalize the value.")
                params.append(_dic)
            else:
                params.append(pickle.load(f))

    # body files
    files = config["bodies"]["files"]
    for f in files:
        if not os.path.exists(f"{Path.cwd()}/{path}/{f}"):
            print(f"Dataset is not valid. {Path.cwd()}/{path}/{f} not found.")
    train_files, train_params, train_names, test_files, test_params, test_names = train_test_split(files, params, train_proportion, seed, shuffle)
    train_files = [ f"{Path.cwd()}/{path}/{f}" for f in train_files]
    test_files = [ f"{Path.cwd()}/{path}/{f}" for f in test_files]
    # gym env
    gym_env_filename = f"{path}/{config['gym_env']['filename']}"
    gym_env_class = config["gym_env"]["class"]
    gym_env_id = config["gym_env"]["env_id"]
    register_env(gym_env_id, gym_env_filename, gym_env_class)

    return dataset_name, gym_env_id, train_files, train_params, train_names, test_files, test_params, test_names

def register_env(gym_env_id, gym_env_filename, gym_env_class):
    if not os.path.exists(gym_env_filename):
        print(f"Dataset env file not found. {gym_env_filename}")
    
    assert gym_env_filename[-3:] == ".py"
    gym_env_filename = gym_env_filename[:-3] # remove ".py"
    gym_env_filename = gym_env_filename.replace("/",".") # turn file path into python class path
    gym.register(id=gym_env_id, entry_point=f"{gym_env_filename}:{gym_env_class}", kwargs={'render': False, 'xml': "", 'param': []})

def train_test_split(data, params, train_proportion=0.8, seed=0, shuffle=True):
    assert 0 < train_proportion < 1
    n_train = int(train_proportion * len(data))
    data = np.array(data)
    params = np.array(params)
    indices = np.arange(start=0, stop=len(data))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_files, test_files = data[indices][:n_train], data[indices][n_train:]
    train_params, test_params = params[indices][:n_train], params[indices][n_train:]
    return train_files, train_params, body_names(train_files), test_files, test_params, body_names(test_files)

def body_names(filenames):
    names = []
    for filename in filenames:
        assert filename[-4:]==".xml"
        filename = filename[:-4]
        tmp = filename.split("/")
        filename = tmp[-1]
        names.append(filename)
    return names

if __name__ == "__main__":
    dataset_name, env_id, train, train_names, test, test_names = load_walker2d()
    print(env_id)
    print(train_names)
    print(test)