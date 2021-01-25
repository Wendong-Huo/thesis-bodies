import time
import sys
import os
import hashlib
import numpy as np
from common.tflogs2pandas import tflog2pandas
from common import ga

# hack argv
if sys.argv[1] == "python":
    sys.argv = sys.argv[2:]
from common import common
args = common.args



str_md5 = hashlib.md5(args.custom_alignment.encode()).hexdigest()

folder = f"output_data/{args.tensorboard}/model-{args.train_bodies_str}-CustomAlignWrapper-md{str_md5}-sd{args.seed}/PPO_1"
if not os.path.exists(folder):
    print("results don't exists! tell manager, please!")
    print(folder)
    exit(1)

df = tflog2pandas(folder)
results = []
for body in args.train_bodies:
    df_one_body = df[df["metric"] == f"eval/{body}_mean_reward"]
    if df_one_body.shape[0]==0:
        print("null result! tell manager, please!")
        print(folder)
        exit(1)
    # doesn't impose weighted sum here
    results.append(df_one_body["value"].max())
print(results)
fitness = np.mean(results)

# To avoid infinite loop submitting many jobs
time.sleep(2)

alignment = None
alignments = args.custom_alignment.split("::")
geno = []
for i,a in enumerate(alignments):
    b = a.split(",")
    alignment = [int(x.strip()) for x in b]
    geno.append(alignment)

my_ga = ga.GA()
test_individual = {
    "id": args.ga_job_id,
    "parents": args.ga_parent_id,
    "fitness": float(fitness)+1,
    "geno": geno,
}
data = my_ga.evoke_master(test_individual)
