import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas
from common import common, colors
args = common.args

exp_name = "803.1.walkerwitharms"
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
g_m0 = {
    "900":  3210.9951171875,
    "901":  3235.283740234375,
    "902":  2915.403466796875,
    "903":  3219.646826171875,
    "904":  2636.1869140625,
    "905":  2682.38720703125,
    "906":  3226.565185546875,
    "907":  2944.790380859375,
} # from 803.1.3.walkerwitharms.results.m0.py
g_cursor = 0

def load_tb(force=0):
    try:
        if force:
            raise FileNotFoundError
        df = pd.read_pickle(cache_path)
    except FileNotFoundError:
        dfs = []
        all_jobs = []
        count_cache = [None] * 100
        str_md5_cache = [None] * 100
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
                match = re.findall(r'9xx_mutate_([0-9]+)\/model-([0-9\-]+)-CustomAlignWrapper-md([0-9a-z]+)-sd([0-9]+)', tb_path)
                if len(match)==0:
                    print("Error")
                    exit(1)
                match = match[0]
                num_mutate = int(match[0])
                str_bodies = match[1]
                str_md5 = match[2]
                run_seed = match[3]
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
                if count_cache[num_mutate] is None:
                    count_cache[num_mutate] = 0
                    str_md5_cache[num_mutate] = {}
                if not str_md5 in str_md5_cache[num_mutate]:
                    count_cache[num_mutate] += 1
                    str_md5_cache[num_mutate][str_md5] = count_cache[num_mutate]
                df["num_mutate"] = num_mutate
                df["str_md5"] = str_md5
                df["run_seed"] = run_seed
                df["label"] = f"# {str_md5_cache[num_mutate][str_md5]}"
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

import seaborn as sns
import matplotlib.pyplot as plt

facet_grid_col_order = ["Walker2d", "Halfcheetah", "Ant", "Hopper"]

def check_finished():
    sns.countplot(data=df[df["step"]==df["step"].max()], x="str_md5") # check every run is here
    plt.show()
    plt.close()
    exit(0)
# check_finished()

def plot_learning_curve(df, title="", filename=""):
    g = sns.FacetGrid(data=df, col="robot_id", row="num_mutate", hue="label", ylim=[0,3500])
    g.map(sns.lineplot, "step", "Learnability")
    g.fig.suptitle(title)

    def _const_line(data, **kwargs):
        robot = data.iloc[0]
        plt.axhline(y=g_m0[robot], color=colors.plot_color[1], linestyle=(0, (1, 5)), linewidth=1)
        plt.locator_params(nbins=3)
    g.map(_const_line, "robot_id")

    g.set_xlabels(label="step")
    g.set_ylabels(label="Learnability")

    plt.tight_layout()
    g.add_legend()
    # g.set(yscale="log")
    plt.savefig(f"{output_path}/{exp_name}{filename}.learning_curve.png")
def pick_best_and_worst(df):
    dfs = []
    for num_mutate in [2,4,8,16,32]:
        _df = df[df["num_mutate"]==num_mutate]
        mean_final_values = _df.groupby(['str_md5'], sort=False)['value'].mean().sort_values()
        worst_body_seed = mean_final_values.index[0]
        best_body_seed = mean_final_values.index[-1]
        _df = df[df["str_md5"]==best_body_seed].copy()
        _df["label"] = "best in 21"
        dfs.append(_df)
        _df = df[df["str_md5"]==worst_body_seed].copy()
        _df["label"] = "worst in 21"
        dfs.append(_df)
    return pd.concat(dfs)
_df = pick_best_and_worst(df)
plot_learning_curve(df=_df, title="Train on 8 topologically different bodies with obvious correspondence\nRandom Search at different permutation distance (n=21)\nLearning curve (N=5)")

def density_at_final_step(df, step, title, filename=""):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) # Warning: sns.distplot will be removed in the future version.

    _df = df[df["step"]==step]
    g = sns.FacetGrid(data=_df, col="num_mutate", ylim=[0,3500])
    g.map(sns.distplot, "Learnability", vertical=True, hist=False, rug=True)
    
    # def _const_line(data, **kwargs):
    #     robot = data.iloc[0]
    #     plt.axhline(y=g_16_selected[robot], color=colors.plot_color[1], linestyle=(0, (1, 5)), linewidth=1)
    #     plt.locator_params(nbins=3)
    # g.map(_const_line, "robot")
    
    g.fig.suptitle(title)
    g.set_xlabels("Density")
    g.set_ylabels("Learnability")
    plt.locator_params(nbins=3)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{exp_name}{filename}.density.png")
density_at_final_step(df=df, step=df["step"].max(), title=f"Train on 8 topologically different bodies with obvious correspondence\nComparison of different permutation distance\nDensity at step {df['step'].max()//1e5/10:.1f}e6 (N=5)")
