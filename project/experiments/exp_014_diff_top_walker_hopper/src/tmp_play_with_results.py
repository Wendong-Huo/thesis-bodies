import pandas as pd
import pickle

# with open("output_data/all_jobs.pickle", "rb") as f:
#     all_jobs = pickle.load(f)
# for job in all_jobs.values():
#     if job['seed']==298:
#         print(job)

# exit(0)
df = pd.read_pickle("output_data/tmp/same_topology_all.pickle")

df = df[df["body"]=="hopper"]
df = df[df["step"]==720896.0]
df = df[df["num_bodies"]==4]
# df = df[df["method"]=="aligned"]
df = df[df["method"]=="general_only"]

print(df)
print(df.shape[0])
print(df["value"].mean())
print(df["value"].std())
print(df["seed"].unique())

df = pd.read_pickle("output_data/tmp/same_topology_oracles")
df = df[df["body"]=="hopper"]
df = df[df["step"]==720896.0]
print(df)
print(df.shape[0])
print(df["value"].mean())
print(df["value"].std())
print(df["seed"].unique())
