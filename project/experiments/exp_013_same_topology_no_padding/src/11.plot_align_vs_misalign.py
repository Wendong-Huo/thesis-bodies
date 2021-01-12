import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_all = pd.read_pickle("output_data/tmp/same_topology_all.pickle")
# smooth oracles by choosing less steps
valid_steps = list(df_all[df_all["method"]=="aligned"]["step"].unique())
df_all = df_all[df_all["step"].isin(valid_steps)]
# remove hopper's feetcontact_only, because hopper only has one feet, randomize that will be exactly the same with aligned.
df_all = df_all[(df_all["body"]!="hopper")|(df_all["method"]!="feetcontact_only")]
df_all["randomization"] = df_all["method"]


for plot_num_bodies in [2,4,8,16]:
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
    plt.savefig(f"output_data/plots/aligned_vs_misaligned_{plot_num_bodies}.png")
    # plt.show()
    plt.close()