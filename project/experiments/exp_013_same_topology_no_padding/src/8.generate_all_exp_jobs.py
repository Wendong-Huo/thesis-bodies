import numpy as np
import pickle

np.random.seed(0)

all_experiment_seeds = np.arange(1000)
np.random.shuffle(all_experiment_seeds)
current_exp_id = 0

job_sh_header = """#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
"""
print(job_sh_header)

all_jobs = {}

def print_job_sh(arr):
    global current_exp_id, all_jobs
    print(f"# Job {current_exp_id} Start")
    arr = sorted(arr)
    body_types = [3,4,5,6]
    for body_type in body_types:
        print(f"bodies_{body_type}00=", end="")
        print( ','.join([f"{body_type}{x:02}" for x in arr]) )

    seed = all_experiment_seeds[current_exp_id]
    print(f"""
for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed={seed}
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done""")
    print(f"# Job {current_exp_id} End\n")
    all_jobs[current_exp_id] = {
        "arr": arr,
        "seed": seed,
    }
    current_exp_id += 1

all_bodies = np.arange(20)

repetition = 5

for i in range(repetition):
    np.random.shuffle(all_bodies)
    print_job_sh(all_bodies[:2])
for i in range(repetition):
    np.random.shuffle(all_bodies)
    print_job_sh(all_bodies[:4])
for i in range(repetition):
    np.random.shuffle(all_bodies)
    print_job_sh(all_bodies[:8])
for i in range(repetition):
    np.random.shuffle(all_bodies)
    print_job_sh(all_bodies[:16])

with open("output_data/all_jobs.pickle", "wb") as f:
    pickle.dump(all_jobs, f)