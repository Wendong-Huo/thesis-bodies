import pandas as pd
import pickle

# with open("output_data/all_jobs.pickle", "rb") as f:
#     all_jobs = pickle.load(f)
# for job in all_jobs.values():
#     if job['seed']==298:
#         print(job)

# exit(0)
df_all = pd.read_pickle("output_data/tmp/same_topology_all.pickle")
# smooth oracles by choosing less steps
valid_steps = list(df_all[df_all["method"]=="aligned"]["step"].unique())
df_all = df_all[df_all["step"].isin(valid_steps)]
# remove hopper's feetcontact_only, because hopper only has one feet, randomize that will be exactly the same with aligned.
df_all = df_all[(df_all["body"]!="hopper")|(df_all["method"]!="feetcontact_only")]
df_all["randomization"] = df_all["method"]

# df_part = df_all[(df_all["body"]=="hopper")&(df_all["num_bodies"]==16)&((df_all["method"]=="aligned")|(df_all["method"]=="feetcontact_only"))&(df_all["step"]==163840)]
# print(df_part.head(40))
# exit(0)
# df_all = df_all[]
# df_all = df[(df["body"]=="ant")&(df["method"]=="aligned")]

# print(df)
# print(df.shape[0])
# print(df["value"].mean())
# print(df["value"].std())
# print(df["seed"].unique())

import seaborn as sns
import matplotlib.pyplot as plt
# sns.lineplot(data=df, x="step", y="value")
# plt.show()

# for plot_num_bodies in [2,4,8,16]:
if True:
    plot_num_bodies = 16
    df_all["aligned"] = "misaligned"
    df_all.loc[df_all["num_bodies"]==1,"aligned"] = "oracle"
    df_all.loc[df_all["method"]=="aligned","aligned"] = "aligned"
    df_part = df_all[(df_all["num_bodies"]==plot_num_bodies)|(df_all["num_bodies"]==1)]
    # df_part = df_all[((df_all["num_bodies"]==plot_num_bodies))&(df_all["body"]=="hopper")&((df_all["method"]=="aligned")|(df_all["method"]=="joints_only"))]
    # df_part = df_all[(df_all["num_bodies"]==plot_num_bodies)]
    print(df_part)
    print("Plotting...")
    g = sns.FacetGrid(df_part, col="body", hue="aligned", legend_out=True)
    g.map(sns.lineplot, "step", "value")
    g.add_legend()
    plt.savefig(f"output_data/plots/tmp_aligned_vs_misaligned.png")
    # plt.show()
    plt.close()
if True:
    plot_num_bodies = 16
    df_part = df_all[((df_all["num_bodies"]==plot_num_bodies)|(df_all["num_bodies"]==1))&((df_all["method"]=="aligned")|(df_all["method"]=="joints_only")|(df_all["method"]=="oracle"))]
    print(df_part)
    print("Plotting...")
    g = sns.FacetGrid(df_part, col="body", hue="randomization", legend_out=True)
    g.map(sns.lineplot, "step", "value")
    g.add_legend()
    plt.savefig(f"output_data/plots/tmp_aligned_vs_ra_joints.png")
    plt.close()
if True:
    plot_num_bodies = 16
    df_part = df_all[((df_all["num_bodies"]==plot_num_bodies)|(df_all["num_bodies"]==1))]
    print(df_part)
    print("Plotting...")
    g = sns.FacetGrid(df_part, col="body", hue="randomization", legend_out=True)
    g.map(sns.lineplot, "step", "value")
    g.add_legend()
    plt.savefig(f"output_data/plots/tmp_aligned_vs_others.png")
    plt.close()