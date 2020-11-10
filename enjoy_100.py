import pickle, sys, os
import numpy as np
import subprocess
import enjoy
import ctypes


with open("read_tb.pickle", "rb") as f:
    (max_body_xss, max_body_x_stepss) = pickle.load(f)

mean_xs = np.mean(max_body_xss, axis=1)
arg = np.argsort(mean_xs)

for i, body_id in enumerate(arg):
    print(f" No. {99-i}. Body {body_id}: \n best test record during training: {max_body_xss[body_id,0]:.02f} {max_body_xss[body_id,1]:.02f} {max_body_xss[body_id,2]:.02f}", file=sys.stderr)
    print("")
    record = enjoy.enjoy(
        stats_path=f"logs_3x100/{body_id}_0/ppo/Walker2Ds-v0_1/Walker2Ds-v0", 
        model_path=f"logs_3x100/{body_id}_0/ppo/Walker2Ds-v0_1/best_model.zip", 
        dataset="dataset/walker2d_v6",
        body_id=body_id, 
        n_timesteps=1000, test_time=1, render=True)
    print(f" this time: {record[0]:.02f}", file=sys.stderr)