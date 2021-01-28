from common import seeds
import numpy as np
import pickle, hashlib

np.random.seed(0)

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
        order = np.arange(8)
        for i in range(4):
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
vanilla_bodies = [399,499,599,699]

def one_exp():
    global g_current_exp_id, g_all_jobs
    str_body_selected = ','.join([str(x) for x in vanilla_bodies])
    assert check_not_in_body_history(str_body_selected), "Not lucky! Not a new set of bodies!"
    print("\n# train on bodies: ", vanilla_bodies)

    for seed in range(40):
        if seed==0:
            # I have a good candidate from exp_019
            custom_alignment = "1,5,0,4,6,7,3,2::3,7,0,6,2,4,5,1::5,4,7,0,1,6,2,3::3,7,5,4,0,6,2,1"
        else:
            custom_alignment = construct_custom_alignment()
        assert check_not_in_alignment_history(custom_alignment), "Not lucky! Not a new alignment!"
        print(f"\n# ==> Alignment {seed} : ", custom_alignment)
        str_md5 = hashlib.md5(custom_alignment.encode()).hexdigest()
        print(f"# {str_md5}")
        with seeds.temp_seed(seed):
            run_seeds = np.random.randint(low=0, high=10000, size=[5])
        for run_seed in run_seeds:
            seed = g_current_exp_id
            print(f"sbatch -J vanilla4 submit.sh python 1.train.py --train_steps=5e6 --seed={run_seed} --train_bodies={str_body_selected} --test_bodies={str_body_selected} --topology_wrapper=CustomAlignWrapper --custom_alignment={custom_alignment} --tensorboard=tensorboard_vanilla4 --custom_align_max_joints=8")

            g_all_jobs[g_current_exp_id] = {
                "vanilla_bodies": vanilla_bodies,
                "custom_alignment": custom_alignment,
                "seed": seed,
                "str_md5": str_md5,
                "run_seed": run_seed,
            }
            g_current_exp_id += 1

for i in range(1):
    one_exp()

with open("output_data/jobs_vanilla4_revi.pickle", "wb") as f:
    pickle.dump(g_all_jobs, f)

print(f"\n# In total: {g_current_exp_id} jobs will be submitted.")