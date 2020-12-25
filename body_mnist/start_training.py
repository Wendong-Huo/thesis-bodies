import os,math,itertools
import yaml
import numpy as np

import utils
args = utils.args

total_sbatch = 0
def sbatch(cmd):
    global total_sbatch
    final_cmd = f"sbatch --job-name={args.exp} submit.sh {cmd}"
    print(final_cmd)
    os.system(final_cmd)
    total_sbatch += 1

def train_one(train, test, withbody, seed=0):
    str_train = ','.join(str(i) for i in train)
    str_test = ','.join(str(i) for i in test)
    sbatch(f"python train.py --exp={args.exp} --seed={seed} --train-bodies={str_train} --test-bodies={str_test} {'--with-bodyinfo' if withbody else ''} ")

def move_test_to_end(all, test):
    assert isinstance(all, list) and isinstance(test, list), "move_test_to_end() only use list operators, don't use np.array"
    for t in test:
        assert t in all, f"body {t} not found."
        all.remove(t)
    all += test

if __name__ == "__main__":
    total_jobs = []

    classA_all = list(range(5))
    classB_all = [x+100 for x in classA_all]
    len_all_comb = int(math.pow(math.comb(5,1), 2)) # ({10 choose 2}^3)


    np.random.seed(0)
    test_choice = np.random.choice(np.arange(0,len_all_comb), size=[25], replace=False)
    test_choice.sort()
    print(test_choice)

    iteration = 0
    for testA in list(itertools.combinations(classA_all,1)):
        for testB in list(itertools.combinations(classB_all,1)):
                if iteration in test_choice: # not do all of them, but only a little fraction of them.
                    classA = classA_all.copy()
                    classB = classB_all.copy()
                    
                    move_test_to_end(classA, list(testA))
                    move_test_to_end(classB, list(testB))

                    print(f"classA {classA}, classB {classB}")

                    test = classA[4:] + classB[4:]
                    
                    # train on test single
                    if False:
                        for test_body in test:
                            sub_exp(test_body, test_body, False)
                    
                    # # train on 4 A's, test on the rest A and one B
                    # train = classA[:4]
                    # sub_exp(train, test, False)
                    # # train on 4 B's, test on the rest B and one A
                    # train = classB[:4]
                    # sub_exp(train, test, False)

                    train = classA[:4] + classB[:4]
                    train_one(train, test, False, seed=iteration)
                    train_one(train, test, True, seed=iteration)

                    one_job = {
                        "train": train,
                        "test": test,
                    }
                    total_jobs.append(one_job)
                iteration += 1
    print(f"total iteration: {iteration}")
    print(f"total sbatch jobs: {total_sbatch}")
    utils.write_yaml(f"{utils.folder}/total_jobs.yml", total_jobs)

    # total_jobs_filename = f"{utils.folder}/total_jobs.yml"
    # with open(total_jobs_filename, "r") as f:
    #     total_jobs = yaml.load(f, Loader=yaml.SafeLoader)

    # for job in total_jobs:
    #     train(job["train"], job["test"], withbody=False, seed=0)
    #     train(job["train"], job["test"], withbody=True, seed=0)
    
    # print(f"total_sbatch: {total_sbatch}")