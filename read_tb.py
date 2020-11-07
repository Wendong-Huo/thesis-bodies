import glob, os
import pickle
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

if os.path.exists("read_tb.pickle"):
    with open("read_tb.pickle", "rb") as f:
        (max_body_xs, max_body_x_steps, arg) = pickle.load(f)
else:
    max_body_xs = []
    max_body_x_steps = []
    for body_id in range(100):
        event_folder=f"tb/{body_id}/Walker2Ds-v0/PPO_1"
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
                    if summary.tag == "debug/body_x":
                        body_x = summary.tensor.float_val[0]
                        if body_x > max_body_x:
                            max_body_x = body_x
                            max_body_x_step = step
            print(f"Peak at step {max_body_x_step}, peak body x: {max_body_x}")
            max_body_xs.append(max_body_x)
            max_body_x_steps.append(max_body_x_step)

    max_body_xs = np.array(max_body_xs)
    max_body_x_steps = np.array(max_body_x_steps)
    arg = np.argsort(max_body_xs)
    with open("read_tb.pickle", "wb") as f:
        pickle.dump((max_body_xs, max_body_x_steps, arg), f)

with open("read_tb.html", "w") as f:
    print("<html><body>", file=f)
    for i, body_id in enumerate(arg):
        print(f"<div class='rank'># {i}", file=f)
        print(f"<div class='title'>Body {body_id}</div>", file=f)
        print(f"<div class='content'>Max Body X: {max_body_xs[body_id]} at step {max_body_x_steps[body_id]}<br><img src='screenshots/body_{body_id}.png'></div>", file=f)
    print("</body></html>", file=f)