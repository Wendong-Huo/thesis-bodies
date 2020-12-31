#!/bin/sh
set -x

for seed in 0 1 2 3 4
do
    for body in 300 400 500 600
    do
        #Control
        sbatch -J exp0 submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=3e6
        #Treatment
        sbatch -J exp0 submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --stack_frames=4 --train_steps=3e6
        sbatch -J exp0 submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --stack_frames=8 --train_steps=3e6
    done
done

