import pickle
import numpy as np
import enjoy
import argparse
from pathlib import Path

def _worker_one(train_body_id, num):
    records_a = []
    for test_body_id in range(num):
        records_b = []
        for run in range(3):
            record = enjoy.enjoy(
                stats_path=f"logs_3x100/{train_body_id}_{run}/ppo/Walker2Ds-v0_1/Walker2Ds-v0", 
                model_path=f"logs_3x100/{train_body_id}_{run}/ppo/Walker2Ds-v0_1/best_model.zip", 
                dataset="dataset/walker2d_v6",
                body_id=test_body_id, 
                n_timesteps=1000, test_time=3, render=False)
            records_b.append(record)
            print(f"{train_body_id}->{test_body_id}, run {run}: distances: {record}")
        records_a.append(records_b)
    return np.array(records_a)


parser = argparse.ArgumentParser()
parser.add_argument("--train", type=int)
# parser.add_argument("--test", type=int)
args = parser.parse_args()
ret = _worker_one(args.train, 100)
print(ret)
print(ret.shape)
Path("misc/cross_test/").mkdir(parents=True, exist_ok=True)
with open(f"misc/cross_test/train_on_{args.train}.pickle", "wb") as f:
    pickle.dump(ret,f)

