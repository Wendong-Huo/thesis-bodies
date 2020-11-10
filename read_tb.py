import glob, os
import pickle
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

if os.path.exists("read_tb.pickle"):
    with open("read_tb.pickle", "rb") as f:
        (max_body_xss, max_body_x_stepss) = pickle.load(f)
else:
    max_body_xss = []
    max_body_x_stepss = []
    for body_id in range(100):
        max_body_xs = []
        max_body_x_steps = []
        for run in range(3):
            event_folder=f"tb_3x100/{body_id}_{run}/Walker2Ds-v0/PPO_1"
            files = glob.glob(f"{event_folder}/*")

            for event_file in files:
                loader = EventFileLoader(event_file)
                print(event_file)

                # Where to store values
                wtimes,steps,actions = [],[],[]
                max_body_x = 0
                max_body_x_step = 0
                for event in loader.Load():
                    wtime   = event.wall_time
                    step    = event.step
                    if len(event.summary.value) > 0:
                        summary = event.summary.value[0]
                        if summary.tag == "eval/body_x":
                            body_x = summary.tensor.float_val[0]
                            if body_x > max_body_x:
                                max_body_x = body_x
                                max_body_x_step = step
                print(f"Peak at step {max_body_x_step}, peak body x: {max_body_x}")
                max_body_xs.append(max_body_x)
                max_body_x_steps.append(max_body_x_step)
        max_body_xss.append(max_body_xs)
        max_body_x_stepss.append(max_body_x_steps)

    max_body_xss = np.array(max_body_xss)
    max_body_x_stepss = np.array(max_body_x_stepss)
    with open("read_tb.pickle", "wb") as f:
        pickle.dump((max_body_xss, max_body_x_stepss), f)

print(max_body_xss.shape)
mean_xs = np.mean(max_body_xss, axis=1)
max_xs = np.max(max_body_xss, axis=1)
min_xs = np.min(max_body_xss, axis=1)
arg = np.argsort(mean_xs)[::-1]

with open("misc/read_tb.html", "w") as f:
    print("<html><link rel='stylesheet' href='styles.css'><body>", file=f)
    for i, body_id in enumerate(arg):
        print(f"<div class='rank'># {i}", file=f)
        print(f"<div class='title'>Body {body_id}</div>", file=f)
        print(f"<div class='content'>Max Body X: <br>", file=f)
        for i in range(3):
            print(f"<big>{max_body_xss[body_id,i]:.01f}</big> at step {max_body_x_stepss[body_id,i]}<br>", file=f)
        print(f"<img src='../screenshots/body_{body_id}.png' width='100'></div>", file=f)
        print(f"</div>", file=f)
    print("</body></html>", file=f)

import matplotlib.pyplot as plt
plt.plot(np.arange(0,100), mean_xs[arg], color='C0')
plt.fill_between(np.arange(0,100), min_xs[arg], max_xs[arg], color='C0', alpha=.1)
plt.xlabel("different bodies, order by mean distance")
plt.ylabel("distance")
plt.title("Different Bodies' Performances (Train and test on same body)")
# plt.show()
plt.savefig("misc/3x100_performance.png")
plt.close()
