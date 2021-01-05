import os
import shutil
import pandas as pd
from common.tflogs2pandas import tflog2pandas, many_logs2pandas
from common import common
args = common.args
threshold = 1500

for body_base in args.train_bodies:
    results = pd.DataFrame(columns=["body", "value", "succeed"])
    for idx in range(50):
        body = body_base+idx
        if args.stack_frames>1:
            path = f"output_data/tensorboard/model-{body}-stack4-sd0/PPO_1"
        else:
            path = f"output_data/tensorboard/model-{body}-sd0/PPO_1"
        if not os.path.exists(path):
            continue
        df = tflog2pandas(path)
        df = df[df["metric"]==f"eval/{body}_mean_reward"]
        if df["value"].max()<threshold:
            continue
        df = df[df["value"]>=threshold]
        succeed = df["step"].min()
        max_value = df["value"].max()
        results = results.append({"body":body, "succeed": succeed, "value":max_value}, ignore_index=True)
        # print(body, max_value)

    results = results.sort_values(by="succeed")
    print(results)
    print("\n"*2)
    print("selected:")
    results = results[results["value"]<10000]
    results = results.iloc[:20]

    selected_bodies = results["body"].to_numpy().tolist()
    print(selected_bodies)
    
    
    if args.stack_frames>1:
        folder = "output_data/selected_bodies_sf"
    else:
        folder = "output_data/selected_bodies"
    os.makedirs(folder, exist_ok=True)

    re_index = {3:0,4:0,5:0,6:0}
    for body in selected_bodies:
        body = int(body)
        body_type = body//100
        shutil.copy(f"output_data/bodies/{body}.xml", f"{folder}/{body_type*100+re_index[body_type]}.xml")
        re_index[body_type] += 1
