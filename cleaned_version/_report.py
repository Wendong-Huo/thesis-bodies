# Author: Sida Liu, 2020
#   Making plots
#
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from arguments import get_args_report
from _report_utils.data import load_multi_data
from utils import output

args = get_args_report()


def figure_1(force_reload=True):
    fname = f"_report_cache/figure_1.pickle"
    if os.path.exists(fname) and not force_reload:
        with open(fname, "rb") as f:
            (data_body, data_nobody) = pickle.load(f)
    else:
        data_body = load_multi_data(args.folder, bodyinfo=True, only_see_not_trained=True, max_measured_steps=3e6)
        data_nobody = load_multi_data(args.folder, bodyinfo=False, only_see_not_trained=True, max_measured_steps=3e6)
        with open(fname, "wb") as f:
            pickle.dump((data_body, data_nobody), f)
    
    valid_value_idx_body = np.where( data_body>0 )
    valid_value_idx_nobody = np.where( data_nobody>0 )
    for i in range(len(valid_value_idx_body)):
        assert np.array_equal(valid_value_idx_body[i], valid_value_idx_nobody[i])
    output(f"{len(valid_value_idx_body[0])} non-zero value.",2)
    mu = np.mean(data_body[valid_value_idx_body], keepdims=False)
    std = np.std(data_body[valid_value_idx_body], keepdims=False)
    output(f"with body) mean: {mu}, std: {std}", 2)
    mu = np.mean(data_nobody[valid_value_idx_nobody], keepdims=False)
    std = np.std(data_nobody[valid_value_idx_nobody], keepdims=False)
    output(f"without body) mean: {mu}, std: {std}", 2)


def main():
    if args.figure_1:
        figure_1()


if __name__ == "__main__":
    main()
