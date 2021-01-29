import pickle,os,glob
import numpy as np
import pandas as pd
from common import tflogs2pandas
from common import common, colors
args = common.args

exp_name = "803.1.walkerwitharms.m0"
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
        count_cache = [None] * 100
        str_md5_cache = [None] * 100
        use_glob = False
        try:
            with open(f"output_data/jobs/{exp_name}.pickle", "rb") as f:
                all_jobs = pickle.load(f)
        except:
            use_glob = True
        if use_glob:
            all_jobs = glob.glob(f"{tensorboard_path}/9xx_mutate_0/*/PPO_1")
            
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
    sns.countplot(data=df[df["step"]==df["step"].max()], x="run_seed") # check every run is here
    plt.show()
    plt.close()
    exit(0)
# check_finished()

def plot_learning_curve(df, title="", filename=""):
    g = sns.FacetGrid(data=df, col="robot_id", row="num_mutate", hue="label", ylim=[0,3500])
    g.map(sns.lineplot, "step", "Learnability")
    g.fig.suptitle(title)

    # def _const_line(data, **kwargs):
    #     robot = data.iloc[0]
    #     plt.axhline(y=g_16_selected[robot], color=colors.plot_color[1], linestyle=(0, (1, 5)), linewidth=1)
    #     plt.locator_params(nbins=3)
    # g.map(_const_line, "robot")

    g.set_xlabels(label="step")
    g.set_ylabels(label="Learnability")

    plt.tight_layout()
    g.add_legend()
    # g.set(yscale="log")
    plt.savefig(f"{output_path}/{exp_name}{filename}.learning_curve.png")
plot_learning_curve(df=df, title="Train on 8 topologically different bodies with obvious correspondence (M0)\nLearning curve (N=5)")

def density_at_final_step(df, step, title, filename=""):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) # Warning: sns.distplot will be removed in the future version.

    _df = df[df["step"]==step]
    g = sns.FacetGrid(data=_df, col="robot_id", ylim=[0,3500])
    g.map(sns.distplot, "Learnability", vertical=True, hist=False, rug=True)
    
    # def _const_line(data, **kwargs):
    #     robot = data.iloc[0]
    #     plt.axhline(y=g_16_selected[robot], color=colors.plot_color[1], linestyle=(0, (1, 5)), linewidth=1)
    #     plt.locator_params(nbins=3)
    # g.map(_const_line, "robot")
    
    def _print(data, **kwargs):
        print("Mean: ", data.mean())
    g.map(_print, "Learnability")

    g.fig.suptitle(title)
    g.set_xlabels("Density")
    g.set_ylabels("Learnability")
    plt.locator_params(nbins=3)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{exp_name}{filename}.density.png")
# density_at_final_step(df=df, step=df["step"].max(), title=f"Train on 8 topologically different bodies with obvious correspondence (M0)\nDensity at step {df['step'].max()//1e5/10:.1f}e6 (N=5)")

print("Mean of all 8 bodies:")
print(df[df["step"]==df["step"].max()]["Learnability"].mean())