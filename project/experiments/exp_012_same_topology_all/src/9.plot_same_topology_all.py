import os, pickle
from common.tflogs2pandas import tflog2pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from common.gym_interface import template

with open("output_data/all_jobs.pickle", "rb") as f:
    all_jobs = pickle.load(f)

def same_topology_get_results(tensorboard_path, body_arr, seed, method, read_cache=False):
    df_results = pd.DataFrame()
    str_body_arr = '-'.join(str(x) for x in body_arr)
    stacked_training_bodies = np.array(body_arr)
    exp = [
        (stacked_training_bodies + 300).tolist(),
        (stacked_training_bodies + 400).tolist(),
        (stacked_training_bodies + 500).tolist(),
        (stacked_training_bodies + 600).tolist(),
    ]
    seeds = [seed]
    need_retrain = []
    try:
        if not read_cache:
            raise Exception("not_read_cache")
        df_results = pd.read_pickle(f"output_data/tmp/same_topology_{str_body_arr}_sd{seed}_{method}")
    except Exception as e:
        if str(e)!="not_read_cache" and not isinstance(e, FileNotFoundError):
            raise e
        for bodies in exp:
            filename = "-".join([str(x) for x in bodies])
            for seed in seeds:
                if method=="joints_only":
                    str_method = "-ra-ph-pfc"
                elif method=="aligned":
                    str_method = "-"
                elif method=="general_joints_feetcontact":
                    str_method = "-ra"
                elif method=="joints_feetcontact":
                    str_method = "-ra-ph"
                else:
                    str_method = f"-{method}"
                path = f"{tensorboard_path}/model-{filename}{str_method}-sd{seed}/PPO_1"
                if not os.path.exists(path):
                    path = f"{tensorboard_path}/model-{filename}-{method}-sd{seed}/PPO_1"
                    if not os.path.exists(path):
                        raise Exception(f"Path not found. {path}")
                print(f"Loading {path}")
                df = tflog2pandas(path)
                for body in bodies:
                    df_body = df[df["metric"]==f"eval/{body}_mean_reward"].copy()
                    if df_body.shape[0]<62:
                        print(df_body.shape)
                        need_retrain.append({
                            "bodies": bodies,
                            "seed": seed,
                            "method": method,
                        })
                        # raise Exception("Data is not complete.")
                    df_body["body"] = template(body)
                    df_body["seed"] = seed
                    df_body["method"] = method
                    df_body["num_bodies"] = len(body_arr)
                    df_results = df_results.append(df_body)

        df_results.to_pickle(f"output_data/tmp/same_topology_{str_body_arr}_sd{seed}_{method}")
        
    return df_results


def same_topology_get_oracles(tensorboard_path, read_cache=False):
    method = "oracles"
    df_results = pd.DataFrame()
    seeds = list(range(2))
    body_types = [
        300,400,500,600
    ]
    num_tested_bodies = 16
    try:
        if not read_cache:
            raise Exception("not_read_cache")
        df_results = pd.read_pickle(f"output_data/tmp/same_topology_{method}")
    except Exception as e:
        if str(e)!="not_read_cache" and not isinstance(e, FileNotFoundError):
            raise e
        for body_type in body_types:
            for tested in range(num_tested_bodies):
                body = body_type + tested
                for seed in seeds:
                    path = f"{tensorboard_path}/model-{body}-sd{seed}/PPO_1"
                    print(f"Loading {path}")
                    df = tflog2pandas(path)
                    df_body = df[df["metric"]==f"eval/{body}_mean_reward"].copy()
                    if df_body.shape[0]!=62:
                        print(df_body.shape)
                        raise Exception("Data is not complete.")
                    df_body["body"] = template(body)
                    df_body["seed"] = seed
                    df_results = df_results.append(df_body)
        df_results["method"] = "oracle"
        df_results["num_bodies"] = 1
        df_results.to_pickle(f"output_data/tmp/same_topology_{method}")
    return df_results


df_all = []

read_cache = False

df = same_topology_get_oracles(tensorboard_path=f"output_data/tensorboard_same_topology_all/oracles", read_cache=True)
df_all.append(df)

methods = ["aligned", "general_only", "joints_only", "feetcontact_only", "general_joints", "general_feetcontact", "joints_feetcontact", "general_joints_feetcontact"]
for job in all_jobs.values():
    print(f"job array: {job['arr']}")
    print(f"seed: {job['seed']}")
    arr = job['arr']
    seed = job['seed']
    # body_types = [3,4,5,6]
    # for body_type in body_types:
    #     str_bodies = '-'.join([f"{body_type}{x:02}" for x in arr])
    #     # print( str_bodies )
    #     path = f"output_data/tensorboard/model-{str_bodies}--sd{seed}"
    #     if os.path.exists(path):
    for method in methods:
        try:
            df = same_topology_get_results(tensorboard_path=f"output_data/tensorboard", method=method, seed=seed, body_arr=arr, read_cache=read_cache)
            df_all.append(df)
        except Exception as e:
            if str(e)=="Data is not complete.":
                print(e)
                continue
        # else:
        #     print(f"missing path: {path}")

df_all = pd.concat(df_all)
df_all.to_pickle("output_data/tmp/same_topology_all.pickle")
g = sns.FacetGrid(df_all, col="body",  hue="method", legend_out=True)
g.map(sns.lineplot, "step", "value")
g.add_legend()
plt.savefig("output_data/plots/plot_same_topology_all.png")
plt.close()

exit(0)

df_all = {}
df_all["oracles"] = same_topology_get_oracles(f"output_data/tensorboard_same_topology_all/oracles", False)

methods = ["aligned", "general_only", "joints_only", "feetcontact_only", "general_joints", "general_feetcontact", "joints_feetcontact", "general_joints_feetcontact"]
for method in methods:
    df_all[method] = same_topology_get_results(f"output_data/tensorboard_same_topology_all/{method}", method, False)
df = pd.concat(df_all.values())
print(df)

g = sns.FacetGrid(df, col="body",  hue="method", legend_out=True)
g.map(sns.lineplot, "step", "value")
g.add_legend()
plt.savefig("output_data/plots/plot_same_topology_all.png")
plt.close()
