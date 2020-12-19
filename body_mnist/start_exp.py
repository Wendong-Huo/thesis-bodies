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
    sbatch(f"python train_simple.py --exp={args.exp} --train-bodies={str_train} --test-bodies={str_test} {'--with-bodyinfo' if bodyinfo else ''} ")

if __name__ == "__main__":

    classA_all = [0,1,2,3,4]
    classB_all = [100,101,102,103,104]
    
    iteration = 0
    for testA in list(itertools.combinations(classA_all,1)):
        for testB in list(itertools.combinations(classB_all,1)):
            if iteration % 9==0: # not do all of them, but only a little fraction of them.
                classA = classA_all.copy()
                classB = classB_all.copy()
                
                classA.remove(testA[0])
                classA += testA
                classB.remove(testB[0])
                classB += testB
                print(f"classA {classA}, classB {classB}")


                test = classA[4:] + classB[4:]
                
                # train on test single
                sub_exp(test[:1], test[:1], False)
                sub_exp(test[1:], test[1:], False)

                # train on 4 A's, test on the rest A and one B
                train = classA[:4]
                sub_exp(train, test, False)
                # train on 4 B's, test on the rest B and one A
                train = classB[:4]
                sub_exp(train, test, False)
                # train on 4 A's and 4 B's, test on the rest A and the rest B
                train = classA[:4] + classB[:4]
                sub_exp(train, test, False)
                # train on 4 A's and 4 B's, test on the rest A and the rest B, with bodyinfo
                train = classA[:4] + classB[:4]
                sub_exp(train, test, True)
            
            iteration += 1

    print(f"total sbatch jobs: {total_sbatch}")