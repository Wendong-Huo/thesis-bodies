#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

# Check significance


for seed in $(seq 0 10)
do
# best_alignment in vanilla4:
    best_alignment=4,3,2,5,0,6,1,7::4,5,6,2,3,1,7,0::0,5,1,3,2,7,6,4::4,0,2,1,7,5,3,6
# 25aca1bfcda9ab49da1adcf14a804d7f

# worst_alignment in vanilla4:
    worst_alignment=6,2,3,0,4,5,7,1::2,1,3,5,6,7,4,0::7,4,0,3,1,2,6,5::6,4,2,5,7,3,0,1
# 31e3558068aa1c4101069f4373b7eafd
    sbatch -J vanilla4_check_sig submit.sh python 1.train.py  --custom_alignment=$best_alignment --train_steps=5e6 --seed=$seed --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --topology_wrapper=CustomAlignWrapper --tensorboard=tensorboard_vanilla4_checks --custom_align_max_joints=8
    sbatch -J vanilla4_check_sig submit.sh python 1.train.py  --custom_alignment=$best_alignment --train_steps=5e6 --seed=$seed --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --topology_wrapper=CustomAlignWrapper --tensorboard=tensorboard_vanilla4_checks --custom_align_max_joints=8
done