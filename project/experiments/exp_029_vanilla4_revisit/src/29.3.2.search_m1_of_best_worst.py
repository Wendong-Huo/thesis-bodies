import copy, hashlib, pickle
import numpy as np

from common import seeds,utils

exp_name = "vanilla4_mga"

print(utils.shell_header(exp_name, "Search M1 of the best and worst found in 29.1"))

def ga_mutate(alignments, n=1, seed=0):
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

good_alignment="4,3,2,5,0,6,1,7::4,5,6,2,3,1,7,0::0,5,1,3,2,7,6,4::4,0,2,1,7,5,3,6"
bad_alignment="6,2,3,0,4,5,7,1::2,1,3,5,6,7,4,0::7,4,0,3,1,2,6,5::6,4,2,5,7,3,0,1"

def get_offspring_alignments(alignment, num_offsprings=2):
    _ind = [[int(y) for y in x.split(",")] for x in alignment.split("::")]
    _offsprings = []
    retry = 0
    for i in range(num_offsprings):
        while True:
            m = ga_mutate(alignments=_ind, n=1, seed=i+retry)
            if m not in _offsprings:
                break
            retry += 1
        _offsprings.append(m)
    ret = []
    for x in _offsprings:
        o = []
        for y in x:
            o.append(','.join([str(z) for z in y]))
        ret.append('::'.join(o))
    return ret

good_mutants = get_offspring_alignments(good_alignment, 20)
bad_mutants = get_offspring_alignments(bad_alignment, 20)

g_current_exp_id = 0
g_all_jobs = {}
def generate_jobs(mutants, label=""):
    global g_current_exp_id, g_all_jobs
    for offspring_alignment in mutants:
        # print(offspring_alignment,run_seed)
        str_md5 = hashlib.md5(offspring_alignment.encode()).hexdigest()
        print(f"\n# {str_md5}")
        print(f"# {offspring_alignment}")
        with seeds.temp_seed(g_current_exp_id):
            run_seeds = np.random.randint(low=0, high=1000000, size=[5])
        for run_seed in run_seeds:
            cmd = f"sbatch -J vanilla4_manual_ga submit.sh python 1.train.py  --custom_alignment={offspring_alignment} --train_steps=5e6 --seed={run_seed} --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --topology_wrapper=CustomAlignWrapper --tensorboard=tensorboard_{exp_name} --custom_align_max_joints=8"
            print(cmd)
            g_all_jobs[g_current_exp_id] = {
                # "vanilla_bodies": vanilla_bodies,
                "custom_alignment": offspring_alignment,
                "seed": g_current_exp_id,
                "str_md5": str_md5,
                "run_seed": run_seed,
                "label": label,
            }
            g_current_exp_id+=1
generate_jobs(good_mutants, label="good")
generate_jobs(bad_mutants, label="bad")
print(f"# In Total {g_current_exp_id} jobs.")

print(utils.shell_tail())

with open(f"output_data/jobs_{exp_name}.pickle", "wb") as f:
    pickle.dump(g_all_jobs, f)
