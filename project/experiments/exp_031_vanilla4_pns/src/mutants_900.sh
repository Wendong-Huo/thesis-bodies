#!/bin/sh

set -x

bodies_900=900,901,902,903,904,905,906,907

for seed in 0 1 2 3 4
do
    # control, doesn't consider alignment
    sbatch -J mut900 submit.sh python 1.train.py --train_bodies=$bodies_900 --test_bodies=$bodies_900 --seed=$seed
    # treatment, align using joint_names
    sbatch -J mut900 submit.sh python 1.train.py --train_bodies=$bodies_900 --test_bodies=$bodies_900 --seed=$seed --topology_wrapper=MutantWrapper
done