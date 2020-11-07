import pickle
import subprocess
with open("read_tb.pickle", "rb") as f:
    (a,b,args) = pickle.load(f)

for i, body_id in enumerate(args):
    print(f" No. {99-i}. Body {body_id}: best record {a[body_id]}")
    print("")
    cmd = ["python", "enjoy.py", f"-f=logs/{body_id}/ppo/Walker2Ds-v0_1/",
                f"--load-checkpoint=logs/{body_id}/ppo/Walker2Ds-v0_1/best_model.zip",
                f"--body-id={body_id}",
                "--dataset=dataset/walker2d_v6",
                "--deterministic"]
    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("")
    # subprocess.call(cmd)