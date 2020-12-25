import os
import yaml

import utils
args = utils.args

total_sbatch = 0
def sbatch(cmd):
    global total_sbatch
    final_cmd = f"sbatch --job-name={args.exp} submit.sh {cmd}"
    print(final_cmd)
    os.system(final_cmd)
    total_sbatch += 1

if __name__ == "__main__":
    for i in range(5):
        sbatch(f"python train.py --exp=exp_add_diff_help --train-bodies=0 --test-bodies=0 --disable-wrapper --seed={i}")
        sbatch(f"python train.py --exp=exp_add_diff_help --train-bodies=0,1 --test-bodies=0 --disable-wrapper --seed={i}")
        sbatch(f"python train.py --exp=exp_add_diff_help --train-bodies=0,2 --test-bodies=0 --disable-wrapper --seed={i}")
        sbatch(f"python train.py --exp=exp_add_diff_help --train-bodies=0,3 --test-bodies=0 --disable-wrapper --seed={i}")
        sbatch(f"python train.py --exp=exp_add_diff_help --train-bodies=0,100 --test-bodies=0 --disable-wrapper --seed={i}")
        sbatch(f"python train.py --exp=exp_add_diff_help --train-bodies=0,101 --test-bodies=0 --disable-wrapper --seed={i}")
        sbatch(f"python train.py --exp=exp_add_diff_help --train-bodies=0,102 --test-bodies=0 --disable-wrapper --seed={i}")
        sbatch(f"python train.py --exp=exp_add_diff_help --train-bodies=0,0,0,0,0,0,0,1 --test-bodies=0 --disable-wrapper --seed={i}")
        sbatch(f"python train.py --exp=exp_add_diff_help --train-bodies=0,0,0,0,0,0,0,100 --test-bodies=0 --disable-wrapper --seed={i}")
    print(f"total_sbatch: {total_sbatch}")