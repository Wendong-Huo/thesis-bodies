from common.tflogs2pandas import tflog2pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from common.gym_interface import template

df_max_values = pd.DataFrame()
df_all_values = pd.DataFrame()

runs = np.arange(start=0, stop=16)
exp = [
    (runs + 300).tolist(),
    (runs + 400).tolist(),
    (runs + 500).tolist(),
    (runs + 600).tolist(),
]
print(exp)
seeds = list(range(10))
methods = ["align", "random"]
need_retrain = []
if True:
    for bodies in exp:
        filename = "-".join([str(x) for x in bodies])
        for method in methods:
            for seed in seeds:
                if method=="align":
                    str_method = ""
                elif method=="misalign":
                    str_method = "-mis"
                elif method=="random":
                    str_method = "-ra"
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

                    
                    df_max_values = df_max_values.append({
                        "body": template(body),
                        "seed": seed,
                        "method": method,
                        "max_value": df_body["value"].max(),
                        "filename": filename,
                    }, ignore_index=True)

                    df_body["body"] = template(body)
                    df_body["seed"] = seed
                    df_body["method"] = method
                    df_body["filename"] = filename
                    df_all_values = df_all_values.append(df_body)


    df_all_values.to_pickle("output_data/tmp/df_all_values")
else:
    df_all_values = pd.read_pickle("output_data/tmp/df_all_values")
print(df_all_values) #[68448 rows x 7 columns]

g = sns.FacetGrid(df_all_values, col="body",  hue="method", legend_out=True)
g.map(sns.lineplot, "step", "value")
g.add_legend()
plt.savefig("output_data/plots/exp_all_values.png")
plt.close()
if False:
    g = sns.FacetGrid(df_max_values, col="body", height=3, aspect=1.5)
    g.map(sns.barplot, "method", "max_value", order=methods)
    g.set_xticklabels(rotation=20)
    plt.savefig("output_data/plots/exp_4x16x10.png")
    plt.close()

    print("need retrain:")
    print(need_retrain)