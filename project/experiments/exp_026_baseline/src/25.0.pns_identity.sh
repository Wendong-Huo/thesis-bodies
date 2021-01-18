#!/bin/sh
# test out  self.electricity_cost = 2.0
# test out  start with identity, learning_rate for pns modules set to 1e-4 
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

for seed in $(seq 0 5)
do
    # Treatment Soft aligned
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init
done

for seed in $(seq 0 5)
do
    # Control Misaligned
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107
done
