#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

for seed in 0 1 2
do
    for body in $(seq 700 749)
    do
        # echo $body
        sbatch -J oracle submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed
    done
    for body in $(seq 800 849)
    do
        # echo $body
        sbatch -J oracle submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed
    done
done