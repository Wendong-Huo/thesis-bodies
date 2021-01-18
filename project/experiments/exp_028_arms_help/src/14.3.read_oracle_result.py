import os, pickle, glob

from pandas.core.reshape.concat import concat
from common.tflogs2pandas import tflog2pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from common.gym_interface import template
if False:
    def read_df(body):
        dfs = []
        for seed in [0,1]:
            folder = f"output_data/tensorboard_oracle_random_bodies/model-{body}-sd{seed}/PPO_1"
            print(f"Loading {folder} ...")
            df = tflog2pandas(folder)
            if df.shape[0]!=1353:
                return None # Fly-away bug causes the job to abort
            df = df[df["metric"]==f"eval/{body}_mean_reward"]
            max_value = df["value"].max()
            final_value = df.iloc[-1, df.columns.get_loc("value")]
            df = pd.DataFrame({
                "body": template(body),
                "body_id": body,
                "max_value": max_value,
                "final_value": final_value,
                "seed": seed,
            }, index=[body])
            dfs.append(df)
        return pd.concat(dfs)

    dfs = []
    for body in np.arange(start=100, stop=200):
        df = read_df(body)
        if df is not None:
            dfs.append(df)

    df = pd.concat(dfs)
    print(df)

    df.to_pickle("output_data/tmp/oracle_1xx_df")

df = pd.read_pickle("output_data/tmp/oracle_1xx_df")
for body_type in [100]:
    df_one_type = df[df["body"]==template(body_type)]
    df_one_type = df_one_type.groupby("body_id").mean()

    df_one_type = df_one_type.sort_values(by="max_value", ascending=False)

    print(df_one_type.head(20))
    selected = df_one_type.head(20).index.tolist()
    start_id = body_type
    for s in selected:
        print(f"cp output_data/bodies/{s}.xml ../input_data/bodies/{start_id}.xml")
        print(f"cp output_data/tmp/model-{s}-sd0.zip output_data/models/model-{start_id}-sd0.zip")
        print(f"cp output_data/tmp/model-{s}-sd1.zip output_data/models/model-{start_id}-sd1.zip")
        start_id += 1
