import numpy as np
import pickle

np.random.seed(0)

body_700s = np.arange(20) + 700
body_800s = np.arange(20) + 800

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


for num_train_bodies in [1,8]:
    for i in range(2):
        body_700s_selected = sorted(np.random.choice(body_700s, size=[num_train_bodies], replace=False))
        body_800s_selected = sorted(np.random.choice(body_800s, size=[num_train_bodies], replace=False))
        str_body_selected = ','.join([str(x) for x in body_700s_selected+body_800s_selected])
        seed = all_experiment_seeds[current_exp_id]
        print(f"# Job {current_exp_id} Start")
        print("#", current_exp_id, body_700s_selected, body_800s_selected)
        print(f"""
            for case_id in 1 2 3
            do
                sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed={seed} --train_bodies={str_body_selected} --test_bodies={str_body_selected} --topologies=diff --wrapper_type=Walker2DHopperCase$case_id
            done
        """)
        print(f"# Job {current_exp_id} End\n")

        all_jobs[current_exp_id] = {
            "body_700s_selected": body_700s_selected,
            "body_800s_selected": body_800s_selected,
            "seed": seed,
        }
        current_exp_id += 1

with open("output_data/jobs_diff_topology.pickle", "wb") as f:
    pickle.dump(all_jobs, f)

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

