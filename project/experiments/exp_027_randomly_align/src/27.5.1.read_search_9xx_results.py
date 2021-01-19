import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas

with open("output_data/tmp/all_jobs.pickle", "rb") as f:
    all_jobs = pickle.load(f)
bodies = np.arange(900,908).tolist()

def load_tb(force=0):
    cache_path = "output_data/tmp/9xx_tb_results.pandas"
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

            tb_path = f"output_data/tensorboard_9xx/9xx_mutate_{job['num_mutate']}/model-900-901-902-903-904-905-906-907-CustomAlignWrapper-md{job['str_md5']}-sd*/PPO_1"
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

df = load_tb(1)

import seaborn as sns
import matplotlib.pyplot as plt

def check_finished():
    sns.countplot(data=df, x="body_seed", hue="num_mutate") # check every run is here
    plt.show()
    plt.close()
check_finished()
# exit(0)

def plot_all():
    g = sns.FacetGrid(data=df, row="num_mutate", col="metric", hue="body_seed")
    g.map(sns.lineplot, "step", "value")
    plt.savefig("output_data/tmp/all.png")
    plt.close()
# plot_all()

# originally evaluate at 1007616.0, and have similar choice. Anyway we only need a good and a bad solution for each num_mutate
# detail differences refer to 27.5.2.0.generate_confirm_sig_exp.py
def plot_best_vs_worst(num_mutate = 4, evaluate_at_step = 2007040.0, dry_run=False):
    ret = {}
    _df = df[(df["step"]==evaluate_at_step)&(df["num_mutate"]==num_mutate)]
    # print(_df)
    mean_final_values = _df.groupby(['body_seed'], sort=False)['value'].mean().sort_values()
    # print(mean_final_values)
    # print(mean_final_values.index[0], mean_final_values.index[-1])
    worst_body_seed = mean_final_values.index[0]
    best_body_seed = mean_final_values.index[-1]

    if not dry_run:
        _df = df[(df["num_mutate"]==num_mutate) & ((df["body_seed"]==worst_body_seed)|(df["body_seed"]==best_body_seed))].copy()
        _df["label"] = (_df["body_seed"]==best_body_seed)
        g = sns.FacetGrid(data=_df, col="metric", hue="label")
        g.map(sns.lineplot, "step", "value")
        g.fig.suptitle(f"best vs worst in m{num_mutate}")
        axes = g.axes.flatten()
        for i in range(8):
            axes[i].set_title(f"Body {900+i}")
        plt.tight_layout()
        plt.savefig(f"output_data/tmp/best_vs_worst_{num_mutate}.png")
        plt.close()

    print("")
    print(f"best_alignment in mutate {num_mutate}:")
    _df = df[(df["body_seed"]==best_body_seed) & (df["num_mutate"]==num_mutate)]
    print(_df.iloc[0]["custom_alignment"])
    print(_df.iloc[0]["str_md5"])
    ret[f"best_in_m{num_mutate}"] = _df.iloc[0]["str_md5"]
    print(f"vacc seed: {_df.iloc[0]['vacc_run_seed']}")
    print(f"worst_alignment in mutate {num_mutate}:")
    _df = df[(df["body_seed"]==worst_body_seed) & (df["num_mutate"]==num_mutate)]
    print(_df.iloc[0]["custom_alignment"])
    print(_df.iloc[0]["str_md5"])
    ret[f"worst_in_m{num_mutate}"] = _df.iloc[0]["str_md5"]
    print(f"vacc seed: {_df.iloc[0]['vacc_run_seed']}")
    return ret

all_ret = {}
for i in [2,4,8,16,32]:
    ret = plot_best_vs_worst(i, evaluate_at_step=1007616, dry_run=True)
    all_ret.update(ret)
print(all_ret)
with open("output_data/tmp/to_confirm.pickle", "wb") as f:
    pickle.dump(all_ret, f)

def gradient_exists(evaluate_at_step=2007040.0):
    _df = df[df["step"]==evaluate_at_step]
    sns.lineplot(data=_df, x="num_mutate", y="value")
    plt.show()
    plt.close
gradient_exists()
# _df=df[df["str_md5"]=="62fe719074fc13bd2543f7f2bb21173b"]
# _df=df[df["vacc_run_seed"]=="1965"]
# # _df=_df[_df["metric"]=="eval/900_mean_reward"]
# print(_df["value"])
# sns.lineplot(data=_df,x="step",y="value",hue="metric")
# plt.show()
# plt.close()

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
