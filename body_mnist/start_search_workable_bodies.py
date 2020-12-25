import os
import arguments
import itertools

args = arguments.get_args()

total_sbatch = 0
def sbatch(cmd):
    global total_sbatch
    final_cmd = f"sbatch --job-name={args.exp} submit.sh {cmd}"
    print(final_cmd)
    os.system(final_cmd)
    total_sbatch += 1

def sub_exp(train, test, bodyinfo):
    str_train = ','.join(str(i) for i in train)
    str_test = ','.join(str(i) for i in test)
    sbatch(f"python train_one.py --exp={args.exp} --train-bodies={str_train} --test-bodies={str_test} {'--with-bodyinfo' if bodyinfo else ''} ")

if __name__ == "__main__":

    for i in range(500):
        sub_exp(train=[i], test=[i], bodyinfo=False)
    print(f"total sbatch jobs: {total_sbatch}")