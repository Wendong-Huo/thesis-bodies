#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
    body=300
    #control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6 --num_venv=1
    #treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6 --num_venv=1 --stack_frames=4
done

