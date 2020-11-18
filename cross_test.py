import pickle
import numpy as np
from yaml import parse
import enjoy
import argparse
from pathlib import Path


def _worker_one(train_body_id, num):
    if args.train_on == "g20":
        exp_folder = "experiment_results/logs_3xg20"
    if args.train_on == "20u":
        exp_folder = "experiment_results/logs_3x20u"
    elif args.train_on == "g10":
        exp_folder = "experiment_results/logs_3xg10"
    else:
        exp_folder = "experiment_results/logs_3x100"
    exp_algo = args.algo

    exp_folder += args.bodyinfo

    records_a = []
    for test_body_id in range(num):
        records_b = []
        for run in range(3):
            record = enjoy.enjoy(
                stats_path=f"{exp_folder}/{train_body_id}_{run}/{exp_algo}/Walker2Ds-v0_1/Walker2Ds-v0", 
                model_path=f"{exp_folder}/{train_body_id}_{run}/{exp_algo}/Walker2Ds-v0_1/best_model.zip", 
                dataset="dataset/walker2d_v6",
                body_id=test_body_id, 
                algo=exp_algo,
                n_timesteps=1000, test_time=3, render=False)
            records_b.append(record)
            print(f"{train_body_id}->{test_body_id}, run {run}: distances: {record}")
        records_a.append(records_b)
    return np.array(records_a)


parser = argparse.ArgumentParser()
parser.add_argument("--train", type=int)
parser.add_argument("--train_on", type=str, default="1", help="[1, g10, g20, 20u]")
parser.add_argument("--bodyinfo", type=str, default="")
parser.add_argument("--algo", type=str, default="ppo")
#g20 is sorted group of size 20
#20u is unsorted (random) group of size 20

# parser.add_argument("--test", type=int)
args = parser.parse_args()
if args.train_on == "g20":
    result_folder = "cross_test_g20"
elif args.train_on == "20u":
    result_folder = "cross_test_20u"
elif args.train_on == "g10":
    result_folder = "cross_test_g10"
elif args.train_on == "1":
    result_folder = "cross_test"
else:
    raise "Train on?"

ret = _worker_one(args.train, 100)
print(ret)
print(ret.shape)

Path("experiment_results/{result_folder}/").mkdir(parents=True, exist_ok=True)
with open(f"experiment_results/{result_folder}/train_on_{args.train}{args.bodyinfo}.pickle", "wb") as f:
    pickle.dump(ret,f)

