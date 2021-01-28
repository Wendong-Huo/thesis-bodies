#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

for seed in 0 1
do
    for body in $(seq 100 199)
    do
        sbatch -J oracle_random_bodies submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --tensorboard=tensorboard_oracle_random_bodies --train_steps=1e6
    done
done
