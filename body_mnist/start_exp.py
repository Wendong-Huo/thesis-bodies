import os
import itertools
import math

import numpy as np

import arguments
import utils
args = arguments.get_args()

total_sbatch = 0
def sbatch(cmd):
    global total_sbatch
    final_cmd = f"sbatch --job-name={args.exp} submit.sh {cmd}"
    print(final_cmd)
    os.system(final_cmd)
    total_sbatch += 1

def sub_exp(train, test, bodyinfo, seed=0):
    str_train = ','.join(str(i) for i in train)
    str_test = ','.join(str(i) for i in test)
    sbatch(f"python train_simple.py --exp={args.exp} --seed={seed} --train-bodies={str_train} --test-bodies={str_test} {'--with-bodyinfo' if bodyinfo else ''} ")

def move_test_to_end(all, test):
    assert isinstance(all, list) and isinstance(test, list), "move_test_to_end() only use list operators, don't use np.array"
    for t in test:
        assert t in all, f"body {t} not found."
        all.remove(t)
    all += test

if __name__ == "__main__":

    total_jobs = []

    classA_all = [0,1,2,3,4,5,6,7,8,9]
    classB_all = [100,101,102,103,104,105,106,107,108,109]
    classC_all = [200,201,202,203,204,205,206,207,208,209]
    
    len_all_comb = int(math.pow(math.comb(10,2), 3)) # ({10 choose 2}^3)

    np.random.seed(0)
    test_choice = np.random.choice(np.arange(0,len_all_comb), size=[20], replace=False)
    test_choice.sort()
    print(test_choice)

    iteration = 0
    for testA in list(itertools.combinations(classA_all,2)):
        for testB in list(itertools.combinations(classB_all,2)):
            for testC in list(itertools.combinations(classC_all,2)):
                if iteration in test_choice: # not do all of them, but only a little fraction of them.
                    classA = classA_all.copy()
                    classB = classB_all.copy()
                    classC = classC_all.copy()
                    
                    move_test_to_end(classA, list(testA))
                    move_test_to_end(classB, list(testB))
                    move_test_to_end(classC, list(testC))

                    print(f"classA {classA}, classB {classB}, classC {classC}")

                    test = classA[8:] + classB[8:] + classC[8:]
                    
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

                    # train on 8 A's, 8 B's, and 8 C's. test on the rest A, the rest B, and the rest C
                    train = classA[:8] + classB[:8] + classC[:8]
                    sub_exp(train, test, False, seed=iteration)
                    sub_exp(train, test, True, seed=iteration)

                    one_job = {
                        "train": train,
                        "test": test,
                    }
                    total_jobs.append(one_job)
                iteration += 1
    print(f"total iteration: {iteration}")
    print(f"total sbatch jobs: {total_sbatch}")
    utils.write_yaml(f"{utils.folder}/total_jobs.yml", total_jobs)