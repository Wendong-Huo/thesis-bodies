import pickle
import numpy as np
import enjoy
import argparse
from pathlib import Path

def _worker_one(args, num):
    if args.bodyinfo:
        folder = "logs_3xg10"
    else:
        folder = "logs_3xg10_nobodyinfo"

    records_a = []
    for test_body_id in range(num):
        records_b = []
        for run in range(3):
            record = enjoy.enjoy(
                stats_path=f"experiment_results/{folder}/{args.train}_{run}/{args.algo}/Walker2Ds-v0_1/Walker2Ds-v0", 
                model_path=f"experiment_results/{folder}/{args.train}_{run}/{args.algo}/Walker2Ds-v0_1/best_model.zip", 
                dataset="dataset/walker2d_v6",
                algo=args.algo,
                body_id=test_body_id, 
                n_timesteps=1000, test_time=3, render=False)
            records_b.append(record)
            print(f"g{args.train}->{test_body_id}, run {run}: distances: {record}")
        records_a.append(records_b)
    return np.array(records_a)


parser = argparse.ArgumentParser()
parser.add_argument("--train", type=int, default="0")
parser.add_argument("--algo", type=str, default="ppo1")
parser.add_argument("--bodyinfo", action="store_true", default=True)
# parser.add_argument("--test", type=int)
args = parser.parse_args()
ret = _worker_one(args, 100)
print(ret)
print(ret.shape)
extname = "_bodyinfo" if args.bodyinfo else ""
Path("experiment_results/cross_test_group/").mkdir(parents=True, exist_ok=True)
with open(f"experiment_results/cross_test_group/train_on_g{args.train}{extname}.pickle", "wb") as f:
    pickle.dump(ret,f)

