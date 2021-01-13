from numpy.lib.twodim_base import tri
from common import seeds
import numpy as np
import pickle

np.random.seed(0)

body_200s = np.arange(20) + 200

all_experiment_seeds = np.arange(1000)
np.random.shuffle(all_experiment_seeds)

job_sh_header = """#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
"""
print(job_sh_header)



g_seed_construct_custom_alignment = 0
def construct_custom_alignment():
    global g_seed_construct_custom_alignment
    g_seed_construct_custom_alignment += 1
    orders = []
    with seeds.temp_seed(g_seed_construct_custom_alignment):
        order = np.arange(10)
        for i in range(16):
            np.random.shuffle(order)
            orders.append(",".join([str(x) for x in order]))
    ret = "::".join(orders)
    return ret

alignment_history = {}
def check_not_in_alignment_history(idx, also_insert=True):
    global alignment_history
    if idx in alignment_history:
        return False
    if also_insert:
        alignment_history[idx] = True
    return True

g_selected_bodies = {}
def check_not_in_body_history(idx, also_insert=True):
    global g_selected_bodies
    if idx in g_selected_bodies:
        return False
    if also_insert:
        g_selected_bodies[idx] = True
    return True

def test_repetition():
    for i in range(10000000):
        if i % 1000 == 0:
            print(".", end=" ", flush=True)
        alignment = construct_custom_alignment(i)
        assert check_not_in_alignment_history(alignment)

g_current_exp_id = 0
g_all_jobs = {}
def one_exp():
    global g_current_exp_id, g_all_jobs
    num_train_bodies = 16
    body_200s_selected = sorted(np.random.choice(body_200s, size=[num_train_bodies], replace=False))
    str_body_selected = ','.join([str(x) for x in body_200s_selected])
    assert check_not_in_body_history(str_body_selected), "Not lucky! Not a new set of bodies!"
    print("\n# selected bodies: ", body_200s_selected)

    for i in range(2):
        custom_alignment = construct_custom_alignment()
        assert check_not_in_alignment_history(custom_alignment), "Not lucky! Not a new alignment!"
        print(f"# alignment {i} : ", custom_alignment)
        # 5 runs for each alignment
        for j in range(5):
            seed = g_current_exp_id
            print(f"sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed={seed} --train_bodies={str_body_selected} --test_bodies={str_body_selected} --topology_wrapper=CustomAlignWrapper --custom_alignment={custom_alignment}")

            g_all_jobs[g_current_exp_id] = {
                "body_200s_selected": body_200s_selected,
                "custom_alignment": custom_alignment,
                "seed": seed,
            }
            g_current_exp_id += 1

for i in range(20):
    one_exp()

with open("output_data/jobs_random_exp.pickle", "wb") as f:
    pickle.dump(g_all_jobs, f)

