from common.tflogs2pandas import tflog2pandas
import pickle
import hashlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from common.gym_interface import template

with open("output_data/jobs_vanilla4.pickle", "rb") as f:
    jobs = pickle.load(f)

read_cache = False

try:
    if not read_cache:
        raise Exception("read raw data")
    vanilla4_results = pd.read_pickle("output_data/tmp/vanilla4_results")
    vanilla4_results_full = pd.read_pickle("output_data/tmp/vanilla4_results_full")
    
except Exception as e:
    results = []
    dfs = []
    for idx, job in jobs.items():
        ret = job.copy()
        str_bodies = "-".join([str(x) for x in job["vanilla_bodies"]])
        str_md5 = hashlib.md5(job["custom_alignment"].encode()).hexdigest()
        seed = job["seed"]
        path = f"output_data/tensorboard_vanilla/model-{str_bodies}-CustomAlignWrapper-md{str_md5}-sd{seed}/PPO_1"

        print(f"Loading {path}")
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
        results.append(ret)
        df["str_md5"] = str_md5
        df["seed"] = seed
        dfs.append(df)

    vanilla4_results = pd.DataFrame(results)
    vanilla4_results.to_pickle("output_data/tmp/vanilla4_results")

    vanilla4_results_full = pd.concat(dfs, ignore_index=True)
    vanilla4_results_full.to_pickle("output_data/tmp/vanilla4_results_full")

bodies = [399,499,599,699]
for body in bodies:
    vanilla4_results[f"rank_{body}"] = vanilla4_results[f"max_value_{body}"].rank()

A = vanilla4_results["max_value_399"]
B = vanilla4_results["max_value_499"]
C = vanilla4_results["max_value_599"]
D = vanilla4_results["max_value_699"]
vanilla4_results["combined_value"] = 2*A + B + C + 0.5*D
vanilla4_results["combined_rank"] = vanilla4_results["combined_value"].rank()
vanilla4_results = vanilla4_results.sort_values(by="combined_rank", ascending=False)
print(vanilla4_results.head(10))
print(vanilla4_results.tail(10))
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True)
axes = axes.flatten()
for ax, body in zip(axes,bodies):
    g = sns.histplot(data=vanilla4_results, x=f"max_value_{body}", ax=ax)
    g.set(xlabel="Max Value")
    g.set_title(template(body))

plt.tight_layout()
plt.savefig("output_data/plots/distribution_vanilla4_500_exps.png")
plt.close()

plt.figure()
g = sns.histplot(data=vanilla4_results, x="combined_value")
g.set(xlabel="Combined Value = 2A + B + C + 0.5D")
plt.savefig("output_data/plots/distribution_vanilla4_combined.png")
plt.close()

print("The best alignment:")
print(vanilla4_results.iloc[0]["custom_alignment"])

print("The worst alignment:")
print(vanilla4_results.iloc[-1]["custom_alignment"])
