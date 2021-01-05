import os
import pandas as pd
from common.tflogs2pandas import tflog2pandas, many_logs2pandas
from common import common
args = common.args
for body_base in args.train_bodies:
    results = pd.DataFrame(columns=["body","value"])
    for idx in range(50):
        body = body_base+idx
        path = f"output_data/tensorboard/model-{body}-stack4-sd0/PPO_1"
        if not os.path.exists(path):
            continue
        df = tflog2pandas(path)
        df = df[df["metric"]==f"eval/{body}_mean_reward"]
        max_value = df["value"].max()
        results = results.append({"body":body, "value":max_value}, ignore_index=True)
        # print(body, max_value)

    results = results.sort_values(by="value")
    print(results)