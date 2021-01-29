import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas
from common import common, colors
args = common.args

exp_name = "803.2.confirm_significance"
tensorboard_path = f"output_data/tensorboard/{exp_name}"
cache_path = f"output_data/cache/{exp_name}"
output_path = f"output_data/plots/"

g_m0 = 3008.907354736328

df_803_1 = pd.read_pickle("output_data/cache/803.1.walkerwitharms")
def pick_best_and_worst(df):
    dfs = []
    for num_mutate in [2,4,8,16,32]:
        _df = df[df["num_mutate"]==num_mutate]
        mean_final_values = _df.groupby(['str_md5'], sort=False)['value'].mean().sort_values()
        for rank, body_seed in enumerate(mean_final_values.index):

            _df = df[df["str_md5"]==body_seed].copy()
            _df["rank"] = rank
            dfs.append(_df)
    return pd.concat(dfs)
df_803_1 = pick_best_and_worst(df_803_1)

def load_tb(force=0):
    try:
        if force:
            raise FileNotFoundError
        df = pd.read_pickle(cache_path)
    except FileNotFoundError:
        dfs = []
        all_jobs = []
        use_glob = False
        try:
            with open(f"output_data/jobs/{exp_name}.pickle", "rb") as f:
                all_jobs = pickle.load(f)
        except:
            use_glob = True
        if use_glob:
            all_jobs = glob.glob(f"{tensorboard_path}/*/*/PPO_1")
            
        for job in all_jobs:
            if use_glob:
                tb_path = job
                import re
                match = re.findall(r'model-([0-9\-]+)-CustomAlignWrapper-md([0-9a-z]+)-sd([0-9]+)', tb_path)
                if len(match)==0:
                    print("Error")
                    exit(1)
                match = match[0]
                # num_mutate = int(match[0])
                # num_mutate = 0
                str_bodies = match[0]
                str_md5 = match[1]
                run_seed = match[2]
                _df = df_803_1[df_803_1["str_md5"]==str_md5]
                num_mutate = int(_df["num_mutate"].iloc[0])
                label = int(_df["rank"].iloc[0])
            else:
                print(job)
                num_mutate = job["num_mutate"]
                str_bodies = "-".join([str(x) for x in np.arange(start=900, stop=908, dtype=np.int)])
                custom_alignment = job["custom_alignment"]
                if custom_alignment=="":
                    method = "aligned"
                else:
                    method = "randomized"
                job['label'] = method
                tb_path = f"{tensorboard_path}/9xx_mutate_{num_mutate}/model-{str_bodies}-CustomAlignWrapper-md{job['str_md5']}-sd{job['run_seed']}/PPO_1"
            print(f"Loading {tb_path}")
            if not os.path.exists(tb_path):
                continue
            df = tflogs2pandas.tflog2pandas(tb_path)
            df = df[df["metric"].str.startswith("eval/")]
            if use_glob:
                df["num_mutate"] = num_mutate
                df["str_md5"] = str_md5
                df["run_seed"] = run_seed
                df["label"] = "# 1" if label>10 else "# 2"
            else:
                df["num_mutate"] = job["num_mutate"]
                df["alignment_id"] = job["seed"]
                df["custom_alignment"] = job["custom_alignment"]
                df["str_md5"] = job["str_md5"]
                df["vacc_run_seed"] = job['run_seed']
                # df["str_bodies"] = str_bodies
            
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_pickle(cache_path)
    # get robot name:
    df["robot_id"] = df["metric"].str.slice(start=5, stop=8)
    df["robot"] = df["robot_id"].apply(lambda x: common.gym_interface.template(int(x)).capitalize())
    df["Learnability"] = df["value"]

    return df

df = load_tb(args.force_read)
print(df)
def detail(df, column):
    print(sorted(df[column].unique()))
    for i in sorted(df[column].unique()):
        print(df[df[column]==i][column].count(), end=", ")
    print("")
detail(df, "robot_id")
detail(df, "num_mutate")
detail(df, "label")
detail(df, "run_seed")

import seaborn as sns
import matplotlib.pyplot as plt

facet_grid_col_order = ["Walker2d", "Halfcheetah", "Ant", "Hopper"]

def check_finished():
    sns.countplot(data=df[df["step"]==df["step"].max()], x="str_md5") # check every run is here
    plt.show()
    plt.close()
    exit(0)
# check_finished()

def learning_curve():
    g = sns.FacetGrid(data=df, col="num_mutate", hue="label", legend_out=True)
    g.map(sns.lineplot, "step", "value")

    def _const_line(data, **kwargs):
        plt.axhline(y=g_m0, color=colors.plot_color[1], linestyle=(0, (1, 5)), linewidth=1)
        plt.locator_params(nbins=3)
    g.map(_const_line, "robot_id")
    
    g.add_legend()
    g.fig.suptitle(f"Confirm search result by independently perform 10 additional runs.")
    g.set_xlabels("step")
    g.set_ylabels("Learnability")
    plt.tight_layout()
    plt.savefig(f"output_data/plots/{exp_name}.learning_curve.png")
    plt.close()
learning_curve()

