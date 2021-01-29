#!/bin/sh
set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

# The best alignment:
best=7,1,0,2,6,3,5,4::0,1,3,2,4,5,7,6::0,5,1,4,3,2,6,7::7,2,4,6,1,3,0,5::7,1,0,2,6,3,5,4::0,1,3,2,4,5,7,6::0,5,1,4,3,2,6,7::7,2,4,6,1,3,0,5::7,1,0,2,6,3,5,4::0,1,3,2,4,5,7,6::0,5,1,4,3,2,6,7::7,2,4,6,1,3,0,5::7,1,0,2,6,3,5,4::0,1,3,2,4,5,7,6::0,5,1,4,3,2,6,7::7,2,4,6,1,3,0,5

# The worst alignment:
worst=0,6,4,1,7,3,2,5::3,4,6,0,1,5,7,2::4,0,5,1,3,2,7,6::7,6,0,1,5,4,2,3::0,6,4,1,7,3,2,5::3,4,6,0,1,5,7,2::4,0,5,1,3,2,7,6::7,6,0,1,5,4,2,3::0,6,4,1,7,3,2,5::3,4,6,0,1,5,7,2::4,0,5,1,3,2,7,6::7,6,0,1,5,4,2,3::0,6,4,1,7,3,2,5::3,4,6,0,1,5,7,2::4,0,5,1,3,2,7,6::7,6,0,1,5,4,2,3

for seed in $(seq 0 19):
do
    sbatch -J vanilla4_confirm submit.sh python 1.train.py --train_steps=1e7 --seed=$seed --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --topology_wrapper=CustomAlignWrapper --custom_alignment=$best --tensorboard=tensorboard_vanilla4_confirm
    sbatch -J vanilla4_confirm submit.sh python 1.train.py --train_steps=1e7 --seed=$seed --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --topology_wrapper=CustomAlignWrapper --custom_alignment=$worst --tensorboard=tensorboard_vanilla4_confirm
done