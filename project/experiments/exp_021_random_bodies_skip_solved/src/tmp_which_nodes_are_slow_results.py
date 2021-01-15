import pandas as pd

with open("output_data/tmp/which_nodes_are_slow.txt", "r") as f:
    grep_results = f.readlines()

for idx, line in enumerate(grep_results):
    if "1785959" in line:
        print(grep_results[idx-1])
        print(line)
        break
# exit(0)
l = len("output_data/tensorboard/")

df_results = pd.read_pickle("output_data/tmp/which_nodes_are_slow")
df_results["node"] = ""
df_results["num_bodies"] = 0
for idx_df, row in df_results.iterrows():
    path = row["path"][l:]
    df_results.at[idx_df, "path"] = path
    df_results.at[idx_df, "num_bodies"] = len(path.split("-"))-3
    node = ""
    for idx, line in enumerate(grep_results):
        if path in line:
            job_id = line[:7]
            if int(job_id)<1785585 or int(job_id)>1786224:
                continue # I started exp_012 several times
            _tmp = grep_results[idx-1].split(":")[-1]
            node = _tmp.split(".")[0]
            break
    if node=="":
        print("not found.")            
    else:
        df_results.at[idx_df, "node"] = node
    

df_results = df_results.sort_values(by="node")
df_results.to_csv("output_data/tmp/who_slow.csv")
# df_results = df_results[df_results["path"].str.len()>90]
# print(sorted(df_results["path"].str.len().unique()))
# print(df_results.shape)

# df_results["node_prefix"] = df_results["node"].str.slice(start=0, stop=5)
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# sns.barplot(data=df_results, x="node_prefix", y="min_fps", ax=ax)
sns.barplot(data=df_results, x="node", y="min_fps", ax=ax)
plt.xticks(rotation=45)
# ax1 = ax.twinx()
# ax.set_ylim(0,350)
# ax1.set_ylim(0,350)
# sns.lineplot(x=[-0.5,df_results.shape[0]], y=[34.7,34.7], color="black", ax=ax1)
plt.show()
df_results = df_results.sort_values(by="min_fps")
print(df_results.iloc[0])

# df_slow = df_results[df_results["min_fps"]<80]
# print(df_slow["node"].unique())
# for node in df_slow["node"].unique():
#     print(df_results[df_results["node"]==node])

# print(df_results.iloc[-1])