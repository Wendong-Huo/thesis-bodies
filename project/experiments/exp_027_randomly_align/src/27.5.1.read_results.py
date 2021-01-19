import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas

with open("output_data/tmp/all_jobs.pickle", "rb") as f:
    all_jobs = pickle.load(f)
cache_path = "output_data/tmp/9xx_tb_results.pandas"
bodies = np.arange(900,908).tolist()

def load_tb():
    try:
        # raise FileNotFoundError
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
                print(f"Loading {tb_path}")
                if not os.path.exists(tb_path):
                    continue
                df = tflogs2pandas.tflog2pandas(tb_path)
                df = df[df["metric"].str.startswith("eval/")]
                df["num_mutate"] = job["num_mutate"]
                df["body_seed"] = job["body_seed"]
                df["custom_alignment"] = job["custom_alignment"]
                df["str_md5"] = job["str_md5"]
                dfs.append(df)
        df = pd.concat(dfs)
        print(df)
        df.to_pickle(cache_path)
    return df

df = load_tb()
print(df)

import seaborn as sns
import matplotlib.pyplot as plt

def check_finished():
    sns.countplot(data=df, x="body_seed", hue="num_mutate") # check every run is here
    plt.show()
    plt.close()

# g = sns.FacetGrid(data=df, row="num_mutate", col="metric", hue="body_seed")
# g.map(sns.lineplot, "step", "value")
# plt.savefig("output_data/tmp/all.png")
# plt.close()

num_mutate = 4
_df = df[(df["step"]==1007616)&(df["num_mutate"]==num_mutate)]
print(_df)
mean_final_values = _df.groupby(['body_seed'], sort=False)['value'].mean().sort_values()
print(mean_final_values)
print(mean_final_values.index[0], mean_final_values.index[-1])
worst_body_seed = mean_final_values.index[0]
best_body_seed = mean_final_values.index[-1]

_df = df[(df["num_mutate"]==num_mutate) & ((df["body_seed"]==worst_body_seed)|(df["body_seed"]==best_body_seed))]
g = sns.FacetGrid(data=_df, col="metric", hue="body_seed")
g.map(sns.lineplot, "step", "value")
plt.savefig("output_data/tmp/best_vs_worst_4.png")
plt.close()

print(f"best_alignment in mutate {num_mutate}:")
_df = df[(df["body_seed"]==best_body_seed) & (df["num_mutate"]==num_mutate)]
print(_df.iloc[0]["custom_alignment"])
print(_df.iloc[0]["str_md5"])
print(f"worst_alignment in mutate {num_mutate}:")
_df = df[(df["body_seed"]==worst_body_seed) & (df["num_mutate"]==num_mutate)]
print(_df.iloc[0]["custom_alignment"])
print(_df.iloc[0]["str_md5"])


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
