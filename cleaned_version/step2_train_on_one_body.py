from genericpath import exists
import time
import os
import shutil
from subprocess import Popen, call

import numpy as np

from arguments import get_args
from utils import output, abort, read_template, write_script, read_yaml, write_yaml

args = get_args()
g_env_id = ""
g_exp_name = ""
g_total_bodies = 0

def train(env_id, dataset_path):
    global g_env_id, g_exp_name, g_total_bodies
    np.random.seed(0)
    g_env_id = env_id
    g_exp_name = f"{env_id}"
    config = read_yaml(f"{dataset_path}/config.yaml")
    g_total_bodies = config["bodies"]["total"]

    output(f"Train on one body: {env_id}", 2)
    output(f"Total number of bodies: {g_total_bodies}", 2)
    # create folder for slurm output
    clean_outputs_folder()

    # 2. Train on multi bodies
    script = create_scripts(mode="multi")
    start_experiment_multi(script)

    # 1. Train on single bodies
    script = create_scripts(mode="single")
    start_experiment_single(script)
    
    
    # 3. Wait for results
    if args.vacc:
        output(f"Optional. Use the following command to monitor process:\n  check_exp --search={g_exp_name}", 1)


def create_scripts(mode="single"):
    template_filename = f"script-templates/submit_{mode}.sh"
    script_template = read_template(template_filename)

    filename = f"scripts/{g_env_id}_submit_{mode}.sh"
    data = {
        "cwd": os.path.abspath(os.getcwd()),
        "partition": args.partition,
        "dataset": f"dataset/{g_env_id}",
    }
    write_script(filename, data, script_template)
    return filename

def clean_outputs_folder():
    # out.slurm
    out_slurm = "out.slurm"
    if os.path.exists(out_slurm):
        shutil.rmtree(out_slurm)
    os.mkdir(out_slurm)

    # outputs
    folder = f"outputs/{g_exp_name}"
    trash = f"trash/"
    if os.path.exists(folder):
        output(f"Overwrite experiment {g_exp_name}. Moving old data to trash.", 0)
        os.makedirs(trash, exist_ok=True)
        shutil.move(folder, f"{trash}/{g_exp_name}_{time.time()}")
        os.makedirs(folder)

def start_experiment_single(script):
    for i in range(g_total_bodies):
        for j_seed in range(2):
            output(f"Starting {script} with body-idx {i} seed {j_seed}", 1)
            if args.vacc:
                bash = "sbatch"
            else:
                bash = "bash"        
            cmd_w = [bash, script, g_exp_name, str(i), str(j_seed), "--with-bodyinfo", f"{args.n_timesteps}"]
            cmd_wo = [bash, script, g_exp_name, str(i), str(j_seed), "", f"{args.n_timesteps}"]
            output(" ".join(cmd_w),2)
            output(" ".join(cmd_wo),2)
            if args.in_parallel:
                Popen(cmd_w)
                Popen(cmd_wo)
            else:
                call(cmd_wo)
                call(cmd_w)

def start_experiment_multi(script):
    percentage_train = 0.8
    for i in range(10): # no matter how many bodies are there in the dataset, we only do random experiment 10 times.
        all_bodies = np.arange(0,g_total_bodies)
        np.random.shuffle(all_bodies)
        train_bodies = all_bodies[:int(percentage_train*g_total_bodies)]
        test_bodies = all_bodies[int(percentage_train*g_total_bodies):]
        
        output(f"train_bodies {train_bodies}", 2)
        output(f"test_bodies {test_bodies}", 2)
        exp_path = f"outputs/{g_exp_name}"
        os.makedirs(exp_path, exist_ok=True)
        data = {
            "train_bodies": train_bodies,
            "test_bodies": test_bodies,
        }
        write_yaml(f"{exp_path}/exp_{i}_bodies.yml", data)

        str_train_bodies = np.array2string(train_bodies, separator=',')[1:-1]

        for j_seed in range(2):
            output(f"Starting {script} with exp-idx {i} seed {j_seed}", 1)
            if args.vacc:
                bash = "sbatch"
            else:
                bash = "bash"        
            cmd_w = [bash, script, g_exp_name, str(i), str_train_bodies, str(j_seed), "--with-bodyinfo", f"{args.n_timesteps}"]
            cmd_wo = [bash, script, g_exp_name, str(i), str_train_bodies, str(j_seed), "", f"{args.n_timesteps}"]
            output(" ".join(cmd_w),2)
            output(" ".join(cmd_wo),2)
            if args.in_parallel:
                Popen(cmd_w)
                Popen(cmd_wo)
            else:
                call(cmd_w)
                call(cmd_wo)

if __name__ == "__main__":
    train("walker2d_10-v0", "dataset/walker2d_10-v0")
