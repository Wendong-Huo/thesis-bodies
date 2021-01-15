from common.tflogs2pandas import tflog2pandas
import pickle,os
import hashlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from common.gym_interface import template

best_alignment="7,1,0,2,6,3,5,4::0,1,3,2,4,5,7,6::0,5,1,4,3,2,6,7::7,2,4,6,1,3,0,5::7,1,0,2,6,3,5,4::0,1,3,2,4,5,7,6::0,5,1,4,3,2,6,7::7,2,4,6,1,3,0,5::7,1,0,2,6,3,5,4::0,1,3,2,4,5,7,6::0,5,1,4,3,2,6,7::7,2,4,6,1,3,0,5::7,1,0,2,6,3,5,4::0,1,3,2,4,5,7,6::0,5,1,4,3,2,6,7::7,2,4,6,1,3,0,5"
worst_alignment="0,6,4,1,7,3,2,5::3,4,6,0,1,5,7,2::4,0,5,1,3,2,7,6::7,6,0,1,5,4,2,3::0,6,4,1,7,3,2,5::3,4,6,0,1,5,7,2::4,0,5,1,3,2,7,6::7,6,0,1,5,4,2,3::0,6,4,1,7,3,2,5::3,4,6,0,1,5,7,2::4,0,5,1,3,2,7,6::7,6,0,1,5,4,2,3::0,6,4,1,7,3,2,5::3,4,6,0,1,5,7,2::4,0,5,1,3,2,7,6::7,6,0,1,5,4,2,3"
bodies = [399,499,599,699]
jobs = [
    {
        "bodies" : bodies,
        "custom_alignment": best_alignment,
        "label": "good",
    },
    {
        "bodies" : bodies,
        "custom_alignment": worst_alignment,
        "label": "bad",
    },
]
read_cache = True
try:
    if not read_cache:
        raise Exception("read raw data")
    vanilla4_confirm = pd.read_pickle("output_data/tmp/vanilla4_confirm")
    
except Exception as e:
    results = []
    dfs = []
    for job in jobs:
        for seed in range(19):
            ret = job.copy()
            str_bodies = "-".join([str(x) for x in job["bodies"]])
            str_md5 = hashlib.md5(job["custom_alignment"].encode()).hexdigest()
            path = f"output_data/tensorboard_vanilla4_confirm/model-{str_bodies}-CustomAlignWrapper-md{str_md5}-sd{seed}/PPO_1"

            print(f"Loading {path}")
            assert os.path.exists(path)
            df = tflog2pandas(path)
            df = df[df["metric"].str.startswith("eval/")]
            max_value_399 = df[df["metric"]=="eval/399_mean_reward"]["value"].max()
            max_value_499 = df[df["metric"]=="eval/499_mean_reward"]["value"].max()
            max_value_599 = df[df["metric"]=="eval/599_mean_reward"]["value"].max()
            max_value_699 = df[df["metric"]=="eval/699_mean_reward"]["value"].max()
            print(max_value_399)
            ret["path"] = path
            ret["max_value_399"] = max_value_399
            ret["max_value_499"] = max_value_499
            ret["max_value_599"] = max_value_599
            ret["max_value_699"] = max_value_699
            ret["label"] = job["label"]
            results.append(ret)
            df["str_md5"] = str_md5
            df["seed"] = seed
            dfs.append(df)

    vanilla4_confirm = pd.DataFrame(results)
    vanilla4_confirm.to_pickle("output_data/tmp/vanilla4_confirm")


A = vanilla4_confirm["max_value_399"]
B = vanilla4_confirm["max_value_499"]
C = vanilla4_confirm["max_value_599"]
D = vanilla4_confirm["max_value_699"]
vanilla4_confirm["combined_value"] = 2*A + B + C + 0.5*D


plt.figure()
g = sns.FacetGrid(vanilla4_confirm, col="label", legend_out=True)
g.map(sns.histplot, "max_value_399", bins=100)
g.add_legend()
plt.savefig("output_data/plots/vanilla4_confirm_detail.png")
plt.close()

plt.figure()
g = sns.barplot(data=vanilla4_confirm, x="label", y="combined_value")
# g.set(xlabel="")
plt.savefig("output_data/plots/vanilla4_confirm.png")
plt.close()
