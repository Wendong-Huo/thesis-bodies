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

def test_one(train, test, test_as_class, withbody, seed=0):
    str_train = ','.join(str(i) for i in train)
    str_test = ','.join(str(i) for i in test)
    sbatch(f"python test.py --exp={args.exp} --seed={seed} --train-bodies={str_train} --test-bodies={str_test} {'--with-bodyinfo' if withbody else ''} --test-as-class={test_as_class} ")
    # sanity check
    str_model_filename = f"{utils.folder}/model-ant-{'-'.join(str(i) for i in train)}.zip"
    assert os.path.exists(str_model_filename), "model file doesn't exist."

if __name__ == "__main__":
    total_jobs_filename = f"{utils.folder}/total_jobs.yml"
    with open(total_jobs_filename, "r") as f:
        total_jobs = yaml.load(f, Loader=yaml.SafeLoader)

    for job in total_jobs:
        for test_body in job["test"]:
            test_one(job["train"], [test_body], test_as_class=0, withbody=False, seed=0)
            test_one(job["train"], [test_body], test_as_class=test_body//100, withbody=True, seed=0)
    
    print(f"total_sbatch: {total_sbatch}")