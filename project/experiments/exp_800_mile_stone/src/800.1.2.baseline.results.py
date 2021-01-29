import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas
from common import common
args = common.args

exp_name = "800.1.baseline"
tensorboard_path = f"output_data/tensorboard/{exp_name}"
cache_path = f"output_data/cache/{exp_name}"
output_path = f"output_data/plots/"

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

def plot_learning_curve(title="Default Plot"):
    g = sns.FacetGrid(data=df, col="robot", col_order=facet_grid_col_order, ylim=[0,3500])
    g.map(sns.lineplot, "step", "Learnability", color="red", linestyle='--')
    g.fig.suptitle(title)
    # g.set(yscale="log")
    plt.tight_layout()
    plt.savefig(f"{output_path}/800.1.2.baseline.learning_curve.png")
plot_learning_curve("Baseline: Train on one body (N=10)")

def density_at_final_step(step, title):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) # Warning: sns.distplot will be removed in the future version.
    _df = df[df["step"]==step]
    g = sns.FacetGrid(data=_df, col="robot", col_order=facet_grid_col_order, ylim=[0,3500])
    g.map(sns.distplot, "Learnability", vertical=True, hist=False, rug=True, color="red")
    def _const_line(data, **kwargs):
        y = data.mean()
        print(f"Baseline: {y}")
        plt.axhline(y=y, color="red", linestyle=(0, (1, 5)), linewidth=1)
        plt.text(0.002, y, 'mean', fontsize=10, va='bottom', ha='left')
        plt.locator_params(nbins=3)
    g.map(_const_line, "Learnability")
    g.fig.suptitle(f"{title}: Density at step {step//1e5/10:.1f}e6 (N=10)")
    g.set_xlabels("Density")
    g.set_ylabels("Learnability")
    plt.tight_layout()
    plt.savefig(f"{output_path}/800.1.2.baseline.density.png")
density_at_final_step(step=df["step"].max(), title="Baseline")