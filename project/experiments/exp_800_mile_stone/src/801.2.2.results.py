# pick 16
# 801.2.1 All 16*4 robots selected (add a mark to 801.1.results).
# 801.2.2 The distribution of learnability per type. (histogram or density plot with y-axis represents the learnability) (add a threshold to show our selection) (add another line to indicate the baseline as a horizontal green dashed line)

# Note: this part I trained them with stack_frame=4
# And select bodies based on how quick they pass a threshold.
# The main goal is to reduce time of the following experiments.

import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas
from common import common, colors
args = common.args

exp_name = "801.2.1.stacked"
tensorboard_path = f"output_data/tensorboard/{exp_name}"
cache_path = f"output_data/cache/{exp_name}"
output_path = f"output_data/plots/"

g_baselines = [1442.1541809082032, 1762.0622802734374, 1642.0044189453124, 2381.102606201172]
g_cursor = 0

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
            all_jobs = glob.glob(f"{tensorboard_path}/*/PPO_1")
            
        for job in all_jobs:
            if use_glob:
                tb_path = job
            else:
                tb_path = f"{tensorboard_path}/model-399-499-599-699-CustomAlignWrapper-md{job['str_md5']}-sd{job['run_seed']}/PPO_1"
            print(f"Loading {tb_path}")
            if not os.path.exists(tb_path):
                continue
            df = tflogs2pandas.tflog2pandas(tb_path)
            df = df[df["metric"].str.startswith("eval/")]
            if not use_glob:
                df["alignment_id"] = job["seed"]
                df["custom_alignment"] = job["custom_alignment"]
                df["str_md5"] = job["str_md5"]
                df["vacc_run_seed"] = job['run_seed']
                df["label"] = job['label']
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

import seaborn as sns
import matplotlib.pyplot as plt

facet_grid_col_order = ["Walker2d", "Halfcheetah", "Ant", "Hopper"]

def check_finished():
    sns.countplot(data=df, x="robot") # check every run is here
    plt.show()
    plt.close()
# check_finished()
# exit(0)


def plot_learning_curve(title="Default Plot"):
    g = sns.FacetGrid(data=df, col="robot", col_order=facet_grid_col_order, ylim=[0,3000])
    g.map(sns.lineplot, "step", "Learnability", color=colors.plot_color[1])
    # def _const_line(data, **kwargs):
    #     plt.axhline(y=1500, color=colors.plot_color[1])
    # g.map(_const_line, "Learnability")
    g.fig.suptitle(title)
    # g.set(yscale="log")
    plt.tight_layout()
    plt.savefig(f"{output_path}/{exp_name}.learning_curve.png")
# plot_learning_curve("Train on one variant (N=1)")

def density_at_final_step(step, title):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) # Warning: sns.distplot will be removed in the future version.
    _df = df[df["step"]==step]
    g = sns.FacetGrid(data=_df, col="robot", col_order=facet_grid_col_order, ylim=[0,3000])
    g.map(sns.distplot, "Learnability", vertical=True, hist=False, rug=True, color=colors.plot_color[1])
    # def _const_line(data, **kwargs):
    #     plt.axhline(y=1500, color=colors.plot_color[1])
    #     plt.locator_params(nbins=3)
    # g.map(_const_line, "Learnability")
    g.fig.suptitle(f"{title}: Density at step {step/1e6:.1f}e6")
    g.set_xlabels("Density")
    g.set_ylabels("Learnability")
    plt.tight_layout()
    plt.savefig(f"{output_path}/{exp_name}.density.png")
# density_at_final_step(step=df["step"].max(), title="Random variation in body parameters")

threshold = 1500
df = df[df["value"]>=threshold]
df = df.groupby(by="robot_id").min("step")
print("Selected")
dfs = []
for i in [3,4,5,6]:
    _df = df[df.index.str.startswith(str(i))].sort_values("step", ascending=True)[:20]
    dfs.append(_df)
df = pd.concat(dfs)
df = df.reset_index()
df["robot"] = df["robot_id"].apply(lambda x: common.gym_interface.template(int(x)).capitalize())
print(df.head(60))

# Copy them to ../input_data/bodies, and re-index them so that 300,400,500,600 is the quickest learner, and then 301,401,501,601, etc. 
