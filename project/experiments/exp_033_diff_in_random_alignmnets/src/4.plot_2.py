from common.tflogs2pandas import tflog2pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from common.gym_interface import template
df_all = pd.DataFrame()

exp = [
    [300,400],
    [400,500],
    [500,600],
    # [300,400,500,600],
]
seeds = [0,1]
methods = ["align", "misalign", "random"] # 0 align, 1 misalign, 2 random

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
                df_body = df[df["metric"]==f"eval/{body}_mean_reward"]
                print(df_body.shape)
                # print(df_body.columns)
                print(body, df_body["value"].max())
                df_all = df_all.append({
                    "body": template(body),
                    "seed": seed,
                    "method": method,
                    "max_value": df_body["value"].max(),
                    "filename": filename,
                }, ignore_index=True)

print(df_all)
sns.boxplot(hue="method", y="max_value", data=df_all, x="body")
plt.savefig("output_data/plots/2.png")
plt.close()

g = sns.FacetGrid(df_all, row="filename", col="body", height=3, aspect=1.5)
g.map(sns.barplot, "method", "max_value", order=methods)
g.set_xticklabels(rotation=20)
plt.savefig("output_data/plots/3.png")
plt.close()
