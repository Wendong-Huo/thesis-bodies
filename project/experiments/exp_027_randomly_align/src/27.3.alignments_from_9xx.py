import time,copy,hashlib,pickle
import numpy as np
from common import gym_interface,seeds,utils

default_order = {
    "arm": 0,
    "arm_left": 1,
    "thigh": 2,
    "leg": 3,
    "foot": 4,
    "thigh_left": 5,
    "leg_left": 6,
    "foot_left": 7,
}
alignments = []
for body_id in np.arange(start=900, stop=908):
    env = gym_interface.make_env(robot_body=body_id, render=False)()
    env.reset()
    custom_order = []
    for part_name, part in env.robot.parts.items():
        if part_name.startswith("link") or part_name == "torso" or part_name == "floor":
            continue
        custom_order.append(default_order[part_name])
    # Don't argsort! We define order in the order way. (argsort will be perform inside custom alignment class.)
    # alignment = np.argsort(custom_order)
    alignments.append(custom_order)

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
str_md5 = hashlib.md5(str_meaninful_alignment.encode()).hexdigest()
print(f"# {str_md5}")
str_md5 = hashlib.md5("::".join([str_meaninful_alignment]*2).encode()).hexdigest()
print(f"# {str_md5}")
print(f"# most meaningful alignment for 9xx: {str_meaninful_alignment}")
num_jobs = 0
all_jobs = []
for num_mutate in [2,4,8,16,32]:
    for seed in range(20):
        new_alignments = ga_mutate(alignments, n=num_mutate, seed=seed)
        print(f"\n# mutate {num_mutate} alignment for 9xx: {get_str_alignments(new_alignments)}")
        with seeds.temp_seed(seed):
            run_seeds = np.random.randint(low=0, high=10000, size=[5])
        custom_alignment = get_str_alignments(new_alignments)
        str_md5 = hashlib.md5(custom_alignment.encode()).hexdigest()
        print(f"# {str_md5}")
        for run_seed in run_seeds:
            cmd = f"sbatch -J search_9xx_mutate_{num_mutate} submit.sh python 1.train.py --seed={run_seed} --custom_alignment={custom_alignment} --tensorboard=tensorboard/9xx_mutate_{num_mutate} --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907"
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
with open("output_data/tmp/all_jobs.pickle", "wb") as f:
    pickle.dump(all_jobs, f)