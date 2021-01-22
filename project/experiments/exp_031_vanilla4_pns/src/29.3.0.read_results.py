import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas
from common import common
args = common.args

with open("output_data/jobs_vanilla4_revi.pickle", "rb") as f:
    all_jobs = pickle.load(f)
cache_path = f"output_data/tmp/vanilla_revisit_cache"

def load_tb(force=0):
    try:
        if force:
            raise FileNotFoundError
        df = pd.read_pickle(cache_path)
    except FileNotFoundError:
        dfs = []
        for idx, job in all_jobs.items():
            tb_path = f"output_data/tensorboard_vanilla4/model-399-499-599-699-CustomAlignWrapper-md{job['str_md5']}-sd{job['run_seed']}/PPO_1"
            # paths = glob.glob(tb_path)
            # for tb_path in paths:
            # _tmp = tb_path.split("sd")[-1]
            # vacc_run_seed = _tmp.split("/")[0] # need to read the run seed from vacc..
            print(f"Loading {tb_path}")
            if not os.path.exists(tb_path):
                continue
            df = tflogs2pandas.tflog2pandas(tb_path)
            df = df[df["metric"].str.startswith("eval/")]
            # df["num_mutate"] = job["num_mutate"]
            df["alignment_id"] = job["seed"]
            df["custom_alignment"] = job["custom_alignment"]
            df["str_md5"] = job["str_md5"]
            df["vacc_run_seed"] = job['run_seed']
            dfs.append(df)
        df = pd.concat(dfs)
        # print(df)
        df.to_pickle(cache_path)
    return df

df = load_tb(args.force_read)
# print(df)
# print(all_jobs)


import seaborn as sns
import matplotlib.pyplot as plt

def check_finished():
    sns.countplot(data=df, x="str_md5") # check every run is here
    plt.show()
    plt.close()
# check_finished()

def label_body(metric):
    body = ""
    if "399" in metric:
        body = "Walker2D"
    elif "499" in metric:
        body = "HalfCheetah"
    elif "599" in metric:
        body = "Ant"
    elif "699" in metric:
        body = "Hopper"
    return body
df["body"] = df.apply(lambda row: label_body(row['metric']), axis=1)


def label_best_worst(str_md5, best_str_md5 = ""):
    if str_md5==best_str_md5:
        return "Best"
    else:
        return "Worst"

def plot_best_vs_worst(evaluate_at_step = 5005312.0, dry_run=False):
    _df = df[(df["step"]==evaluate_at_step)]
    mean_final_values = _df.groupby(['str_md5'], sort=False)['value'].mean().sort_values()
    # print(mean_final_values)
    # print(mean_final_values.index[0], mean_final_values.index[-1])
    worst_str_md5 = mean_final_values.index[0]
    best_str_md5 = mean_final_values.index[-1]

    print(df[df["str_md5"]==worst_str_md5].head(n=10))

    if not dry_run:
        _df = df[(df["str_md5"]==worst_str_md5)|(df["str_md5"]==best_str_md5)]
        _df['label'] = _df.apply(lambda row: label_best_worst(row['str_md5'], best_str_md5), axis=1)
        g = sns.FacetGrid(data=_df, col="body", hue="label")
        g.map(sns.lineplot, "step", "value")
        g.fig.suptitle("Random generate 40 alignments and pick the best and worst.")
        g.add_legend()
        plt.tight_layout()
        plt.savefig(f"output_data/tmp/best_vs_worst_vanilla4.png")
        plt.close()

    print("")
    print(f"best_alignment:")
    _df = df[(df["str_md5"]==best_str_md5)]
    print(_df.iloc[0]["custom_alignment"])
    print(_df.iloc[0]["str_md5"])
    print(f"worst_alignment:")
    _df = df[(df["str_md5"]==worst_str_md5)]
    print(_df.iloc[0]["custom_alignment"])
    print(_df.iloc[0]["str_md5"])
plot_best_vs_worst(dry_run=False)

def plot_all():
    g = sns.FacetGrid(data=df, col="metric", hue="str_md5")
    g.map(sns.lineplot, "step", "value")
    plt.show()
# plot_all()