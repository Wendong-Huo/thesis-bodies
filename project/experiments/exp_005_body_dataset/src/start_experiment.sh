#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

for seed in 0 1
do
    for body in 300 400 500 600
    do
        #treatment
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6
    done
done

