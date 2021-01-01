import matplotlib.pyplot as plt
from numpy.core.shape_base import stack
import seaborn as sns
import pandas as pd

from common.tflogs2pandas import tflog2pandas, many_logs2pandas
from common.gym_interface import template

bodies = [300,400,500,600]
str_bodies = '-'.join([str(x) for x in bodies])
all_seeds = list(range(20))
all_stackframe = [0,4,8,16]

cache_filename = "output_data/tmp/plot0"
try:
    df = pd.read_pickle(cache_filename)
except:
    dfs = []
    for body in bodies:
        for seed in all_seeds:
            for stackframe in all_stackframe:
                path = f"output_data/tensorboard/model-{str_bodies}"
                if stackframe>0:
                    path += f"-stack{stackframe}"
                path += f"-sd{seed}/PPO_1"
                print(f"Loading {path}")
                df = tflog2pandas(path)
                df["body"] = body
                df["seed"] = seed
                df["stackframe"] = stackframe
                df = df[df["metric"] == f"eval/{body}_mean_reward"]
                print(df.shape)
                # print(df.head())
                dfs.append(df)
        
    df = pd.concat(dfs)
    df.to_pickle(cache_filename)
    print(df.shape)

fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=[10,10])

axes = axes.flatten()
for idx, body in enumerate(bodies):
    sns.lineplot(
        ax=axes[idx],
        data=df[df["body"]==body],
        x="step", y="value", hue="stackframe", style="stackframe",
        markers=True, dashes=False
    ).set_title(f"Train on 4 bodies, test on {template(body)}.")

# plt.legend()
plt.tight_layout()
plt.savefig("output_data/plots/train4test4.png")
# plt.show()