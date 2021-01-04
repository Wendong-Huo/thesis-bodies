import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from common.tflogs2pandas import tflog2pandas, many_logs2pandas
from common.gym_interface import template

bodies = [300]
all_seeds = list(range(20))
all_stackframe = [0,4]

cache_filename = "output_data/tmp/plot0"
try:
    df = pd.read_pickle(cache_filename)
except:
# if True:
    dfs = []
    for body in bodies:
        for seed in all_seeds:
            for stackframe in all_stackframe:
                path = f"output_data/tensorboard/model-{body}"
                if stackframe>0:
                    path += f"-stack{stackframe}"
                path += f"-sd{seed}/SAC_1"
                print(f"Loading {path}")
                if not os.path.exists(path):
                    continue
                df = tflog2pandas(path)
                df["body"] = body
                df["seed"] = seed
                df["stackframe"] = stackframe
                df = df[df["metric"] == f"eval/{body}_mean_reward"]
                print(df.shape)
                print(df.head())
                dfs.append(df)
        
    df = pd.concat(dfs)
    df.to_pickle(cache_filename)
    print(df.shape)
# df = df[::100]
print(df[df["seed"]==0].head())
print(df[df["seed"]==1].head())
print(df[df["seed"]==2].head())
print(df[df["seed"]==3].head())
df1 = pd.DataFrame(columns=df.columns)
print(df1)
for body in bodies:
    for seed in all_seeds:
        for stackframe in all_stackframe:
            df2 = df[(df["body"]==body) & (df["seed"]==seed) & (df["stackframe"]==stackframe)]
            print(df2.shape)
            x = df2.iloc[df2["value"].argsort().iloc[-1]]
            df1 = df1.append(x)
            # for i in range(30):
            if False:
                step_number = 60000
                x = df2.iloc[(df2["step"] - step_number).abs().argsort()[0]]
                if abs(x["step"]-step_number)>1500:
                    print("no")
                else:
                    # print(x)
                    x = x.copy()
                    # x["step"] = step_number
                    df1 = df1.append(x)
df1 = df1[df1["step"]>550000]
print(df1)

print("control")
df2 = df1[df1["stackframe"]==0]
print(f"{df2['value'].mean():.03f} +- {2*df2['value'].std():.03f}")
print("treatment: stackframe")
df2 = df1[df1["stackframe"]==4]
print(f"{df2['value'].mean():.03f} +- {2*df2['value'].std():.03f}")

print(df1.shape, df.shape)
df = df1
fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=[10,10])
sns.barplot(ax=axes, data=df1, x="stackframe", y="value")
# axes = [axes]
# axes = axes.flatten()
# for idx, body in enumerate(bodies):
#     sns.lineplot(
#         ax=axes[idx],
#         data=df[df["body"]==body],
#         x="step", y="value", hue="stackframe",
#         markers=True, dashes=False
#     ).set_title(template(body))

plt.legend()
plt.tight_layout()
plt.savefig("output_data/plots/0.png")
# plt.show()