import pandas as pd
import pickle

# with open("output_data/all_jobs.pickle", "rb") as f:
#     all_jobs = pickle.load(f)
# for job in all_jobs.values():
#     if job['seed']==298:
#         print(job)

# exit(0)
df = pd.read_pickle("output_data/tmp/diff_topology_walkerhopper.pickle")
print(df.head(20))
