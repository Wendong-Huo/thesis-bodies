import pandas as pd
from common.tflogs2pandas import tflog2pandas

import glob

df_results = pd.DataFrame()
filenames = glob.glob("output_data/tensorboard/model-*/PPO_1")
for filename in filenames:
    print(filename)
    df = tflog2pandas(filename)
    df = df[df["metric"]=="time/fps"]
    average_fps = df["value"].mean()
    min_fps = df["value"].min()
    print("average_fps: ", average_fps, ", min_fps: ", min_fps,)
    df_results = df_results.append({
        "path": filename,
        "average_fps": average_fps,
        "min_fps": min_fps,
    }, ignore_index=True)

df_results.to_pickle("output_data/tmp/which_nodes_are_slow")