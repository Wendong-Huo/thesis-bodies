import pickle
from common import common
import numpy as np
exp_name = "PermutationFromGradientVanilla4"
all_jobs = []
common.shell_header(exp_name, "Use pns and pns_init, so we can have gradient for pns weights, we regulate the weight, align them to permutation matrix when needed.")
np.random.seed(0)
common_argument = " ".join(["--train_bodies=399,499,599,699", "--test_bodies=399,499,599,699", "--pns", "--pns_init",
                            "--topology_wrapper=CustomAlignWrapper", "--custom_align_max_joints=8", "--train_steps=5e6", f"--tensorboard=tensorboard/{exp_name}"])

seeds = np.random.randint(low=0, high=100000, size=[20])
for seed in seeds:
    cmd = f"sbatch -J {exp_name} submit.sh python 1.train.py --seed={seed} {common_argument}"
    print(cmd)
    job = {
        "seed": seed,
    }

common.shell_tail()
with open(f"output_data/tmp/all_jobs_{exp_name}.pickle", "wb") as f:
    pickle.dump(all_jobs, f)
