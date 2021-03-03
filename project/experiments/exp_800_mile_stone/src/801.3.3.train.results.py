import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas
from common import common, colors
args = common.args

exp_name = "801.3.train"
tensorboard_path = f"output_data/tensorboard/{exp_name}"
cache_path = f"output_data/cache/{exp_name}"
output_path = f"output_data/plots/"

g_baselines = [1442.1541809082032, 1762.0622802734374, 1642.0044189453124, 2381.102606201172] # from 800.1.2.results
g_baselines = {
    "Walker2d": 1442.1541809082032, 
    "Halfcheetah": 1762.0622802734374, 
    "Ant": 1642.0044189453124, 
    "Hopper": 2381.102606201172
    } # from 800.1.2.results
g_16_selected = {
    "Walker2d": 1695.436912536621,
    "Halfcheetah": 3158.242273966471,
    "Ant": 1961.7465718587239,
    "Hopper": 2531.1571884155273,
}
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
                print(job)
                train_on_bodies = job["train_on_bodies"]
                str_bodies = "-".join([str(x) for x in train_on_bodies])
                custom_alignment = job["custom_alignment"]
                if custom_alignment=="":
                    method = "aligned"
                else:
                    method = "randomized"
                job['label'] = method
                num_bodies = len(train_on_bodies)
                tb_path = f"{tensorboard_path}/{num_bodies}_{method}/model-{str_bodies}-CustomAlignWrapper-md{job['str_md5']}-sd{job['run_seed']}/PPO_1"
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
                df["str_bodies"] = str_bodies
                df["label"] = method
                df["num_bodies"] = num_bodies
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
    g = sns.FacetGrid(data=df, col="robot", row="num_bodies", hue="label", col_order=facet_grid_col_order, ylim=[0,3500])
    g.map(sns.lineplot, "step", "Learnability")
    # g.fig.suptitle(title)

    def _const_line(data, **kwargs):
        robot = data.iloc[0]
        plt.axhline(y=g_16_selected[robot], color=colors.plot_color[1], linestyle=(0, (1, 5)), linewidth=1)
        plt.locator_params(nbins=3)
    g.map(_const_line, "robot")

    g.set_xlabels(label="step")
    g.set_ylabels(label="Episodic Reward")

    plt.tight_layout()
    g.add_legend()
    # g.set(yscale="log")
    plt.savefig(f"{output_path}/{exp_name}{filename}.learning_curve.pdf")
# plot_learning_curve(df=df, title="Train on multiple parametrically different bodies of same topology\naligned vs randomized (N=10)")

def density_at_final_step(df, step, title, filename=""):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) # Warning: sns.distplot will be removed in the future version.

    _df = df[df["step"]==step]
    g = sns.FacetGrid(data=_df, col="robot", row="num_bodies", hue="label", col_order=facet_grid_col_order, ylim=[0,3500])
    g.map(sns.distplot, "Learnability", vertical=True, hist=False, rug=True)
    
    def _const_line(data, **kwargs):
        robot = data.iloc[0]
        plt.axhline(y=g_16_selected[robot], color=colors.plot_color[1], linestyle=(0, (1, 5)), linewidth=1)
        plt.locator_params(nbins=3)
    g.map(_const_line, "robot")

    # g.fig.suptitle(title)
    g.set_xlabels("Density")
    g.set_ylabels("Episodic Reward")
    plt.tight_layout()
    g.add_legend()

    plt.savefig(f"{output_path}/{exp_name}{filename}.density.pdf")
density_at_final_step(df=df, step=df["step"].max(), title=f"Train on multiple parametrically different bodies of same topology\nDensity at step {df['step'].max()//1e5/10:.1f}e6 (N=10)")
