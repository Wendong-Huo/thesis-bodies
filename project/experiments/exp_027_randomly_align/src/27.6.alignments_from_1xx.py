import time,copy,hashlib,pickle
import numpy as np
from common import gym_interface,seeds,utils

alignments = np.stack([np.arange(start=0,stop=8)]*8).tolist()

def ga_mutate(alignments, n=2, seed=0):
    # Mutate individual
    # Swap Mutation: https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
    # small n means less mutation and less randomness
    offspring = copy.deepcopy(alignments)
    with seeds.temp_seed(seed):
        target_sequence_ids = np.random.randint(low=0, high=len(offspring), size=[n])
        target_positions = []
        for i in range(n):
            target_pos = np.random.choice(np.arange(start=0, stop=len(offspring[0])), size=2)
            target_positions.append(target_pos)
    for i in range(n):
        offspring[target_sequence_ids[i]][target_positions[i][0]], offspring[target_sequence_ids[i]][target_positions[i][1]] = \
            offspring[target_sequence_ids[i]][target_positions[i][1]], offspring[target_sequence_ids[i]][target_positions[i][0]]
    return offspring


def get_str_alignments(alignments):
    _aligns = []
    for a in alignments:
        _align = ",".join(str(x) for x in a)
        _aligns.append(_align)
    str_alignments = "::".join(_aligns)
    return str_alignments

print(utils.shell_header())
str_meaninful_alignment = get_str_alignments(alignments)
print(f"# default alignment for 1xx: {str_meaninful_alignment}")
str_md5 = hashlib.md5(str_meaninful_alignment.encode()).hexdigest()
print(f"# {str_md5}")
str_md5 = hashlib.md5("::".join([str_meaninful_alignment]*2).encode()).hexdigest()
print(f"# {str_md5}")
num_jobs = 0
all_jobs = []
for num_mutate in [64]:
    for seed in range(40):
        new_alignments = ga_mutate(alignments, n=num_mutate, seed=seed)
        custom_alignment = get_str_alignments(new_alignments)
        print(f"\n# mutate {num_mutate} alignment for 1xx: {custom_alignment}")
        str_md5 = hashlib.md5(custom_alignment.encode()).hexdigest()
        print(f"# {str_md5}")
        with seeds.temp_seed(seed):
            run_seeds = np.random.randint(low=0, high=10000, size=[5])
        for run_seed in run_seeds:
            cmd = f"sbatch -J search_1xx_mutate_{num_mutate} submit.sh python 1.train.py --seed={run_seed} --custom_alignment={custom_alignment} --tensorboard=tensorboard/1xx_mutate_{num_mutate}_rerun --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --test_steps=1e7"
            print(cmd)
            job = {
                "id": num_jobs,
                "num_mutate": num_mutate,
                "body_seed": seed,
                "str_md5": str_md5,
                "custom_alignment": custom_alignment,
                "run_seed": run_seed,
            }
            all_jobs.append(job)
            num_jobs += 1
print(f"# Total jobs: {num_jobs}.")
with open("output_data/tmp/all_jobs_1xx.pickle", "wb") as f:
    pickle.dump(all_jobs, f)