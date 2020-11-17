from pathlib import Path

import numpy as np
import yaml
import torch.nn as nn
import gym

def get_sorted_ids():
    """ Return a group of body ids: 0 the best group, 9 the worst group """
    import pickle
    with open(f"experiment_results/read_tb_3x100.pickle", "rb") as f:
        (max_body_xss, max_body_x_stepss) = pickle.load(f)
    mean_xs = np.mean(max_body_xss, axis=1)
    arg = np.argsort(mean_xs)[::-1]
    assert(arg.shape[0]==100)
    return arg
