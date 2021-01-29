import pickle,os,glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from common import tflogs2pandas

with open("output_data/tmp/to_confirm.pickle", "rb") as f:
    all_str_md5 = pickle.load(f) # This list is produced in 27.5.1.read_search_9xx_results.py, in plot_best_vs_worst().

cache_path = "output_data/tmp/9xx_tb_confirm_results.pandas"
bodies = np.arange(900,908).tolist()

def load_tb(force=0):
    try:
        if force:
            raise FileNotFoundError
        df = pd.read_pickle(cache_path)
    except FileNotFoundError:
        dfs = []
        for name, str_md5 in all_str_md5.items():
            for run_seed in range(10):
                tb_path = f"output_data/tensorboard_9xx_confirm/9xx_mutate_confirm/model-900-901-902-903-904-905-906-907-CustomAlignWrapper-md{str_md5}-sd{run_seed}/PPO_1"
                print(f"Loading {tb_path}")
                _df = tflogs2pandas.tflog2pandas(tb_path)
                _df = _df[_df["metric"].str.startswith("eval/")]
                _tmp = name.split("_")
                _df["exp"] = _tmp[-1]
                _df["alignment"] = _tmp[0]
                _df["name"] = name
                _df["str_md5"] = str_md5
                _df["run_seed"] = run_seed
                dfs.append(_df)
        df = pd.concat(dfs)
        df.to_pickle(cache_path)
    return df

df = load_tb(1)
print(df)
def check_finished():
    sns.countplot(data=df, x="name") # check every run is here
    plt.show()
    plt.close()
# check_finished()

def learning_curve():
    g = sns.FacetGrid(data=df, col="exp", hue="alignment", legend_out=True)
    g.map(sns.lineplot, "step", "value")
    g.add_legend()
    g.fig.suptitle(f"Confirm search result by independently perform 10 additional runs.")
    plt.tight_layout()
    # plt.show()
    plt.savefig("output_data/plots/9xx_confirm.png")
    plt.close()
learning_curve()

