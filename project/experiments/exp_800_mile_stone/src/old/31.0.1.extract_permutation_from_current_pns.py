import pickle
from stable_baselines3 import PPO
from common import common
from common import pns
import numpy as np
exp_name = "PossibleOrder"
common.shell_header(exp_name, "Extract permutation from successful PNS, see if it will beat 10 random controls.")
all_jobs = []

args = common.args
args.model_filename = "output_data/tmp/best_model.zip"
model = PPO.load(args.model_filename)
# print(model)
orders = []
orders_reverse = []
for i in range(8):
    # print(f"weight[{i}]")
    weights = model.policy.features_extractor.pns[i].weight.detach().numpy()
    weights_p = pns.permutation_matrix(weights)
    # print(weights_p)
    order = np.dot(weights_p, list(range(weights.shape[0])))
    order_reverse = np.argsort(order)
    orders.append(','.join([str(int(x)) for x in order]))
    orders_reverse.append(','.join([str(int(x)) for x in order_reverse]))
    # These two orders happen to be the same, because there is only one swap. But I am not sure which order is what we need.
    # print(order)
    # print("===========")

possible_order = [None]*2
possible_order[0] = '::'.join(orders)
possible_order[1] = '::'.join(orders_reverse)
print("")
print("")
print("")

np.random.seed(0)

common_argument = f"--train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7"

# Treatment possible good alignments
for i in range(2):
    str_md5 = common.md5(possible_order[i])
    print(f"\n# {str_md5}")
    seeds = np.random.randint(low=0,high=10000,size=[5])
    for seed in seeds:
        cmd = f"sbatch -J {exp_name} submit.sh python 1.train.py --seed={seed} --custom_alignment={possible_order[i]} --tensorboard=tensorboard/{exp_name}/good {common_argument}"
        print(cmd)
        job = {
            "seed": seed,
            "custom_alignment": possible_order[i],
            "str_md5": str_md5,
            "label": f"treatment_{i}",
        }
        all_jobs.append(job)

# Random control
random_order = []
for i in range(10):
    orders = []
    for body in range(8):
        order = np.random.permutation(range(12))
        orders.append(','.join([str(int(x)) for x in order]))
    random_order.append('::'.join(orders))
for i in range(10):
    str_md5 = common.md5(random_order[i])
    print(f"\n# {str_md5}")
    seed = np.random.randint(low=0,high=10000)
    cmd = f"sbatch -J {exp_name} submit.sh python 1.train.py --seed={seed} --custom_alignment={random_order[i]} --tensorboard=tensorboard/{exp_name}/random {common_argument}"
    print(cmd)
    job = {
        "seed": seed,
        "custom_alignment": random_order[i],
        "str_md5": str_md5,
        "label": f"treatment_{i}",
    }
    all_jobs.append(job)

common.shell_tail()
with open(f"output_data/tmp/all_jobs_{exp_name}.pickle", "wb") as f:
    pickle.dump(all_jobs,f)