import pickle
from stable_baselines3 import PPO
from common import common
from common import pns
import numpy as np
exp_name = "PermutationFromGradientGPU"
all_jobs = []
common.shell_header(exp_name, "Use pns and pns_init, so we can have gradient for pns weights, we regulate the weight, align them to permutation matrix when needed.")
np.random.seed(1) # CPU version used 0.
common_argument = f"--train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/{exp_name}"

seeds = np.random.randint(low=0, high=100000, size=[5])
for seed in seeds:
    cmd = f"sbatch -J {exp_name} submit-gpu.sh python 1.train.py --seed={seed} {common_argument}"
    print(cmd)
    job = {
        "seed": seed,
    }

common.shell_tail()
with open(f"output_data/tmp/all_jobs_{exp_name}.pickle", "wb") as f:
    pickle.dump(all_jobs,f)