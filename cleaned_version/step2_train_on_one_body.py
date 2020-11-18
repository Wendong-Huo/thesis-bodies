from genericpath import exists
import time
import os
import shutil
import subprocess
from arguments import get_args
from utils import output, abort, read_template, write_script

args = get_args()
g_env_id = ""


def train(env_id, dataset_path):
    global g_env_id
    g_env_id = env_id
    output(f"Train on one body: {env_id}", 2)
    os.makedirs("out.slurm", exist_ok=True)
    # 1. Create scripts
    script = create_scripts()
    # 2. Run scripts in parallel or in series
    clean_outputs_folder()
    start_experiment(script)
    # 3. Wait for results

    pass


def create_scripts():
    template_filename = f"script-templates/submit.sh"
    script_template = read_template(template_filename)

    filename = f"scripts/{g_env_id}_submit.sh"
    data = {
        "cwd": os.path.abspath(os.getcwd()),
        "partition": "bluemoon",
        "dataset": f"dataset/{g_env_id}",
    }
    write_script(filename, data, script_template)
    return filename

def clean_outputs_folder():
    folder = f"outputs/{args.exp_name}"
    trash = f"trash/"
    if os.path.exists(folder):
        output(f"Overwrite experiment {args.exp_name}. Moving old data to trash.", 0)
        os.makedirs(trash, exist_ok=True)
        shutil.move(folder, f"{trash}/{args.exp_name}_{time.time()}")
        os.makedirs(folder)

def start_experiment(script):
    if args.vacc:
        for i in range(10):
            output(f"Starting {script} {i}", 2)
            cmd = ["sbatch", script, args.exp_name, str(i)]
            subprocess.call(cmd)
    else:
        for i in range(10):
            output(f"Starting {script} {i}", 2)
            cmd = ["bash", script, args.exp_name, str(i)]
            subprocess.call(cmd)


if __name__ == "__main__":
    train("walker2d_10-v0", "dataset/walker2d_10-v0")
