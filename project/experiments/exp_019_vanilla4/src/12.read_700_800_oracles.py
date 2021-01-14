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
        for seed in [0,1,2]:
            folder = f"output_data/tensorboard_oracle/model-{body}-caseWalker2DHopperWrapper-sd{seed}/PPO_1"
            print(f"Loading {folder} ...")
            df = tflog2pandas(folder)
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
    for body in np.arange(start=700, stop=750):
        df = read_df(body)
        dfs.append(df)
    for body in np.arange(start=800, stop=850):
        df = read_df(body)
        dfs.append(df)
    df = pd.concat(dfs)
    print(df)

    df.to_pickle("output_data/tmp/oracle_700_800_df")

df = pd.read_pickle("output_data/tmp/oracle_700_800_df")
for body_type in [700,800]:
    df_one_type = df[df["body"]==template(body_type)]
    df_one_type = df_one_type.groupby("body_id").mean()

    df_one_type = df_one_type.sort_values(by="max_value", ascending=False)

    # print(df_one_type.head(20))
    selected = df_one_type.head(20).index.tolist()
    start_id = body_type
    for s in selected:
        print(f"cp output_data/bodies/{s}.xml ../input_data/bodies/{start_id}.xml")
        start_id += 1
