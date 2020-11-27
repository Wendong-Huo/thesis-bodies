import glob
import re
import yaml
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from utils import output, read_yaml


def load_multi_data(folder, bodyinfo=True, only_see_not_trained=True, max_measured_steps=1e6):
    """
    return data in shape [num-exp, num-seed, num-bodies]
    """
    output(f"Scanning folder {folder}.", 2)
    ret = read_yaml(f"{folder}/config.yml")
    data = np.zeros(shape=[int(ret["num-exp"]), int(ret["num-seed"]), int(ret["num-bodies"])])
    exps = list(range(int(ret["num-exp"])))
    # seeds = list(range(int(ret["num-seed"])))
    bodies = list(range(int(ret["num-bodies"])))
    for exp_id in exps:
        exp_yaml = read_yaml(f"{folder}/exp_multi_{exp_id}_bodies.yml")
        if only_see_not_trained:
            query_bodies = exp_yaml["not_train_bodies"]
        else:
            query_bodies = exp_yaml["test_bodies"]
        output(f"query bodies: {query_bodies}.", 2)
        _body = "_body" if bodyinfo else ""
        tb_folders = glob.glob(f"{folder}/tb/multi{_body}/i{exp_id}_s*/PPO_1")
        for tb_folder in tb_folders:
            m = re.search(r"_s([0-9]+)/PPO", tb_folder)
            seed = int(m[1]) % 100
            max_body_xs = read_tb(tb_folder, query_bodies, max_measured_steps)
            output(f"# results: {len(max_body_xs)}",2)
            if len(max_body_xs)==0:
                continue # don't record events with not enough steps.
            for body_idx in max_body_xs:
                body_x = max_body_xs[body_idx]
                data[exp_id, seed, int(body_idx)] = body_x
                output(f"body_idx: {body_idx}, body_x: {body_x}", 2)
            
    return data


def read_tb(event_folder, query_bodies, max_measured_steps=1e6):
    files = glob.glob(f"{event_folder}/*")
    if not 0 < len(files) < 2:
        output(f"Event folder doesn't contain one file. {event_folder}", -1)
        return None
    event_file = files[0]
    output(f"Load {event_file}", 2)
    loader = EventFileLoader(event_file)
    step = -1
    max_body_xs = {}
    max_body_xs_step = {}
    for event in loader.Load():
        # wtime   = event.wall_time
        step = event.step
        if step > max_measured_steps:
            break
        if len(event.summary.value) > 0:
            summary = event.summary.value[0]
            m = re.search(r"eval\/e([0-9]+)_body_x", summary.tag)
            if m:
                body_id = int(m[1])
                if body_id not in query_bodies:
                    continue
                body_x = summary.tensor.float_val[0]
                if body_id not in max_body_xs or body_x > max_body_xs[body_id]:
                    max_body_xs[body_id] = body_x
                    max_body_xs_step[body_id] = step
    output(f"Last event: Step {step}.", 2)
    if step<max_measured_steps:
        output(f"Event doesn't contain enough steps!", -1)
        return []
    return max_body_xs
