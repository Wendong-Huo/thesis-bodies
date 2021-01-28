import pickle, hashlib, os
import numpy as np
from common import common
from common import seeds, generate_exp, gym_interface

exp_name = "801.3.train"
exp_description = """
For each type,
we randomly pick two robots, and train a RL policy on them, with the observation and action space aligned, we measure the learnability at 2e6 steps.
for the same two robots, we train another RL policy on them, with the observation and action space randomized, we measure the learnability at 2e6 steps.
We repeat the two experiments 10 times, so we have a comparison between aligned and randomized when training on 2 robots.
We then repeat the same procedures on 4 robots, 8 robots, 16 robots, and have comparisons for those conditions.
"""
common.shell_header(exp_name, exp_description)

g_all_jobs = []
g_ge = generate_exp.GenerateExp()
def one_exp(num_bodies=2, pick_bodies_from=[], method="aligned", seed=0, repeat=10):
    global g_all_jobs, g_ge
    max_num_joints = gym_interface.get_max_num_joints(pick_bodies_from[:1])
    with seeds.temp_seed(seed):
        run_seeds = np.random.randint(low=0, high=10000, size=[repeat])
    for run_seed in run_seeds:

        with seeds.temp_seed(run_seed):
            train_on_bodies = np.random.choice(pick_bodies_from, size=[num_bodies], replace=False)
        print("\n# train on bodies: ", train_on_bodies)
        str_body_selected = ','.join([str(x) for x in train_on_bodies])
        custom_alignment = "" # default is aligned
        if method=="randomized":
            custom_alignment = g_ge.construct_random_alignment(num_bodies=num_bodies, max_joints=max_num_joints, seed=run_seed)
            assert g_ge.check_not_in_alignment_history(custom_alignment), "Not lucky! Not a new alignment!"
        print(f"\n# ==> Alignment : ", custom_alignment)
        str_md5 = hashlib.md5(custom_alignment.encode()).hexdigest()
        print(f"# {str_md5}")

        print(f"sbatch -J $exp_name $submit_to python 1.train.py --seed={run_seed} -f=$exp_name/{num_bodies}_{method}/ --train_bodies={str_body_selected} --topology_wrapper=CustomAlignWrapper --custom_alignment={custom_alignment} --custom_align_max_joints={max_num_joints}")

        job = {
            "train_on_bodies": train_on_bodies,
            "seed": seed,
            "custom_alignment": custom_alignment,
            "str_md5": str_md5,
            "run_seed": run_seed,
        }
        g_all_jobs.append(job) 


print("#", end="")
seed = 0
for num_bodies in [2,4,8,16]:
    for method in ["aligned", "randomized"]:
        for base_body in [300,400,500,600]:
            bodies = np.arange(base_body, base_body+16)
            one_exp(num_bodies=num_bodies, method=method, pick_bodies_from=bodies, seed=seed, repeat=10)
            seed+=1
print("#")

with open(f"output_data/jobs/{exp_name}.pickle", "wb") as f:
    pickle.dump(g_all_jobs, f)

print(f"\n# In total: {len(g_all_jobs)} jobs will be submitted.")
common.shell_tail()
