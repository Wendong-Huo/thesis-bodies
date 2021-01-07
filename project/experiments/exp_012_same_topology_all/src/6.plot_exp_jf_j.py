from common.tflogs2pandas import tflog2pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from common.gym_interface import template

# df_oracle = pd.read_pickle("output_data/tmp/df_oracle")
def ph_values():
    df_ph_values = pd.DataFrame()

    stacked_training_bodies = np.arange(start=0, stop=16)
    exp = [
        (stacked_training_bodies + 300).tolist(),
        (stacked_training_bodies + 400).tolist(),
        (stacked_training_bodies + 500).tolist(),
        (stacked_training_bodies + 600).tolist(),
    ]
    print(exp)
    seeds = list(range(10))
    methods = ["joints_feet", "joints"]
    need_retrain = []
    if False:
        for bodies in exp:
            filename = "-".join([str(x) for x in bodies])
            for method in methods:
                for seed in seeds:
                    if method=="joints":
                        str_method = "-ra-ph-pfc"
                    elif method=="joints_feet":
                        str_method = "-ra-ph"
                    path = f"output_data/tensorboard/model-{filename}{str_method}-sd{seed}/PPO_1"
                    print(f"Loading {path}")
                    df = tflog2pandas(path)
                    for body in bodies:
                        df_body = df[df["metric"]==f"eval/{body}_mean_reward"].copy()
                        if df_body.shape[0]!=62:
                            print(df_body.shape)
                            need_retrain.append({
                                "bodies": bodies,
                                "seed": seed,
                                "method": method,
                            })
                            break

                        df_body["body"] = template(body)
                        df_body["seed"] = seed
                        df_body["method"] = method
                        # df_body["filename"] = filename
                        df_ph_values = df_ph_values.append(df_body)


        df_ph_values.to_pickle("output_data/tmp/df_ph_values")
    else:
        df_ph_values = pd.read_pickle("output_data/tmp/df_ph_values")
    print(df_ph_values)
    return df_ph_values
df_all_values = pd.read_pickle("output_data/tmp/df_all_values")
df_oracle = pd.read_pickle("output_data/tmp/df_oracle")
df_ph = ph_values()
df = pd.concat((df_all_values,df_oracle,df_ph))

g = sns.FacetGrid(df, col="body",  hue="method", legend_out=True)
g.map(sns.lineplot, "step", "value")
g.add_legend()
plt.savefig("output_data/plots/exp_all_values.png")
plt.close()
