#!/bin/sh
# test out  self.electricity_cost = 2.0 
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

for seed in $(seq 0 5)
do
    # Treatment Soft aligned
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns_energe --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns_energe --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init

    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns_4_energe --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --learning_rate=1e-4
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns_4_energe --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --learning_rate=1e-4
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns_3_energe --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --learning_rate=1e-3
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns_3_energe --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --learning_rate=1e-3
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns_2_energe --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --learning_rate=1e-2
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns_2_energe --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --learning_rate=1e-2
done

for seed in $(seq 0 5)
do
    # Control Misaligned
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --tensorboard=tensorboard_pns_energe --train_steps=1e7 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107
done
