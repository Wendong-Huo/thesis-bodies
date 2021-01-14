#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

for seed in 0 1
do
    for body in 399 499 599 699
    do
        sbatch -J oracle_4vanilla submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --tensorboard=tensorboard_4types --topology_wrapper=CustomAlignWrapper --custom_alignment=0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7
    done
    body=399,499,599,699
    sbatch -J oracle_4vanilla submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --tensorboard=tensorboard_4types --topology_wrapper=CustomAlignWrapper --custom_alignment=0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7
done