import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas

with open("output_data/tmp/all_jobs_1xx.pickle", "rb") as f:
    all_jobs = pickle.load(f)
cache_path = "output_data/tmp/1xx_tb_results.pandas"
bodies = np.arange(100,108).tolist()

def load_tb(force=0):
    try:
        if force:
            raise FileNotFoundError
        df = pd.read_pickle(cache_path)
    except FileNotFoundError:
        visited = {}
        dfs = []
        for job in all_jobs:
            if job['str_md5'] in visited: # because I failed to seed while generating run seed, so I don't know what the run seeds are used.. :( need to use glob to find out.
                continue
            visited[job['str_md5']] = True

            tb_path = f"output_data/tensorboard_1xx/1xx_mutate_{job['num_mutate']}/model-100-101-102-103-104-105-106-107-CustomAlignWrapper-md{job['str_md5']}-sd*/PPO_1"
            paths = glob.glob(tb_path)
            for tb_path in paths:
                _tmp = tb_path.split("sd")[-1]
                vacc_run_seed = _tmp.split("/")[0] # need to read the run seed from vacc..
                print(f"Loading {tb_path}")
                if not os.path.exists(tb_path):
                    continue
                df = tflogs2pandas.tflog2pandas(tb_path)
                df = df[df["metric"].str.startswith("eval/")]
                df["num_mutate"] = job["num_mutate"]
                df["body_seed"] = job["body_seed"]
                df["custom_alignment"] = job["custom_alignment"]
                df["str_md5"] = job["str_md5"]
                df["vacc_run_seed"] = vacc_run_seed
                dfs.append(df)
        df = pd.concat(dfs)
        print(df)
        df.to_pickle(cache_path)
    return df

df = load_tb(0)
print(df)

import seaborn as sns
import matplotlib.pyplot as plt

def check_finished():
    sns.countplot(data=df, x="body_seed", hue="num_mutate") # check every run is here
    plt.show()
    plt.close()
# check_finished() # There are some Fly-away bug! exclude these bodies.
def get_unfinished():
    _count = df.groupby("metric").count()
    # _count = _count[(_count["value"]!=160)&(_count["value"]!=160)]
    print(_count)

    return
    row = 0
    n = 0
    for i in range(40):
        for j in range(5):
            for k in range(8):
                _row = _df.iloc[row]
                assert _row["body_seed"]==i
                _tmp = str(_row["metric"])
                if _tmp == f"eval/10{k}_mean_reward":
                    row+=1
                    print(f"{_row} ..ok")
                else:
                    print(f"{_tmp} ..bad")
                    return
                n+=1
    print(_df)
get_unfinished()
exit(0)
def exclude_bodies_with_fly_away_bug(df):
    _df = df.groupby("body_seed").count()
    _df = _df[_df["value"]==800] # only consider bodies that have 800 records. (Fly-away bug will abort the training early)
    valid_body_seed = _df.index.values
    df = df[df["body_seed"].isin(valid_body_seed)]
    # print(df.head())
    print(f"valid alignment {valid_body_seed} ({len(valid_body_seed)} in total)")
    return df
df = exclude_bodies_with_fly_away_bug(df)

def plot_all():
    g = sns.FacetGrid(data=df, row="num_mutate", col="metric", hue="body_seed")
    g.map(sns.lineplot, "step", "value")
    plt.savefig("output_data/tmp/all.png")
    plt.close()
# plot_all()

def plot_best_vs_worst(num_mutate = 4, evaluate_at_step = 2007040.0, dry_run=False):
    # _df = df[(df["step"]==evaluate_at_step)&(df["num_mutate"]==num_mutate)]
    _df = df[(df["num_mutate"]==num_mutate)]
    # print(_df)
    mean_final_values = _df.groupby(['body_seed'], sort=False)['value'].mean().sort_values()
    # print(mean_final_values)
    # print(mean_final_values.index[0], mean_final_values.index[-1])
    worst_body_seed = mean_final_values.index[0]
    best_body_seed = mean_final_values.index[-1]

    if not dry_run:
        _df = df[(df["num_mutate"]==num_mutate) & ((df["body_seed"]==worst_body_seed)|(df["body_seed"]==best_body_seed))]
        g = sns.FacetGrid(data=_df, col="metric", hue="body_seed")
        g.map(sns.lineplot, "step", "value")
        plt.savefig(f"output_data/tmp/best_vs_worst_1xx_{num_mutate}.png")
        plt.close()

    print("")
    print(f"best_alignment in mutate {num_mutate}:")
    _df = df[(df["body_seed"]==best_body_seed) & (df["num_mutate"]==num_mutate)]
    print(_df.iloc[0]["custom_alignment"])
    print(_df.iloc[0]["str_md5"])
    print(f"worst_alignment in mutate {num_mutate}:")
    _df = df[(df["body_seed"]==worst_body_seed) & (df["num_mutate"]==num_mutate)]
    print(_df.iloc[0]["custom_alignment"])
    print(_df.iloc[0]["str_md5"])
for i in [64]:
    plot_best_vs_worst(i, dry_run=False)

# df_one_body = df[df["metric"]=="eval/904_mean_reward"]
# # sns.lineplot(data=df_one_body, x="step", y="value", hue="body_seed")
# # plt.savefig("output_data/tmp/904.png")
# # plt.close()

# for body_seed in range(20):
#     _df = df_one_body[df_one_body["body_seed"]==body_seed]
#     print(body_seed, _df["value"].max())
# df_one_body_two_alignment = df_one_body[(df_one_body["body_seed"]==0)|(df_one_body["body_seed"]==5)]
# sns.lineplot(data=df_one_body_two_alignment, x="step", y="value", hue="body_seed")
# plt.savefig("output_data/tmp/904_two_alignments.png")
# plt.close()
