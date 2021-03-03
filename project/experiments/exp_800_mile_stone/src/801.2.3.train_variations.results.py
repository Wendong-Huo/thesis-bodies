# here the data are only 20 selected variants plus a original
import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas
from common import common, colors
args = common.args

exp_name = "801.2.train_variations"
tensorboard_path = f"output_data/tensorboard/{exp_name}"
cache_path = f"output_data/cache/{exp_name}"
output_path = f"output_data/plots/"

g_baselines = [1442.1541809082032, 1762.0622802734374, 1642.0044189453124, 2381.102606201172] # from 800.1.2.results
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
df = df[df["robot_id"].apply(lambda x: int(x)%100!=99)]
print(df)
print(sorted(df["robot_id"].unique()))

import seaborn as sns
import matplotlib.pyplot as plt

facet_grid_col_order = ["Walker2d", "Halfcheetah", "Ant", "Hopper"]

def check_finished():
    sns.countplot(data=df, x="robot") # check every run is here
    plt.show()
    plt.close()
    exit(0)
# check_finished()

def plot_learning_curve(df, title="", filename=""):
    g = sns.FacetGrid(data=df, col="robot", col_order=facet_grid_col_order, ylim=[0,3500])
    g.map(sns.lineplot, "step", "Learnability", color=colors.plot_color[1], linestyle='--')
    # g.fig.suptitle(title)
    # g.set(yscale="log")
    g.set_ylabels("Episodic Reward")
    plt.tight_layout()
    plt.savefig(f"{output_path}/{exp_name}{filename}.learning_curve.pdf")
plot_learning_curve(df=df, title="All parametrical variants: Train on one body (N=3)")

def density_at_final_step(df, step, title, filename=""):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) # Warning: sns.distplot will be removed in the future version.
    global g_cursor
    g_cursor = 0

    _df = df[df["step"]==step]
    g = sns.FacetGrid(data=_df, col="robot", col_order=facet_grid_col_order, ylim=[0,3500])
    g.map(sns.distplot, "Learnability", vertical=True, hist=False, rug=True, color=colors.plot_color[1])
    def _const_line(data, **kwargs):
        global g_cursor
        y = data.mean()
        print(f"Mean learnability: {y}")
        plt.axhline(y=y, color=colors.plot_color[1], linestyle=(0, (1, 5)), linewidth=1)
        plt.axhline(y=g_baselines[g_cursor], color="red", linestyle=(0, (1, 5)), linewidth=1)
        g_cursor+=1
        plt.locator_params(nbins=3)
    g.map(_const_line, "Learnability")
    # g.fig.suptitle(title)
    g.set_xlabels("Density")
    g.set_ylabels("Episodic Reward")
    plt.tight_layout()
    plt.savefig(f"{output_path}/{exp_name}{filename}.density.pdf")
density_at_final_step(df=df, step=df["step"].max(), title=f"All parametrical variants: Density at step {df['step'].max()//1e5/10:.1f}e6 (N=3)")

# here the data are only 20 selected variants plus a original, so no need to select
def select_top(df,num_selected=20):
    _df = df[df["step"]==df["step"].max()] # select the records for at the final step
    _df = _df.groupby("robot_id").mean().reset_index() # take average of all 3 seeds
    _df["robot"] = _df["robot_id"].apply(lambda x: common.gym_interface.template(int(x)).capitalize()) # bring back "robot" column
    dfs = []
    for r in facet_grid_col_order:
        _df_1 = _df[_df["robot"]==r] # select one type
        _df_1 = _df_1.sort_values("value", ascending=False) # sort by learnability
        _df_1 = _df_1[:num_selected] # select top n=20
        selected_ids = _df_1["robot_id"]
        dfs.append(df[df["robot_id"].isin(selected_ids)]) # add to outcome
    return pd.concat(dfs)
num_selected = 16 # a detail here is if we select top 20 from all 50
df_selected = select_top(df=df, num_selected=num_selected)
print(df_selected.shape) # n x 4types x 3seeds
density_at_final_step(df=df_selected, step=df["step"].max(), title=f"Selected {num_selected}x4 variants: Density at step {df['step'].max()//1e5/10:.1f}e6 (N=3)", filename="_selected")
