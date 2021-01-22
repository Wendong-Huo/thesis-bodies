#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
# ========================================

exp_name="PossibleOrder"

description="Extract permutation from successful PNS, see if it will beat 10 random controls."






# 1c5b2efc982ba716b99bb5765383024a
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=2732 --custom_alignment=0,1,2,3,4,5,7,6,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,9,6,7,8,5,10,11::0,11,2,3,4,5,6,7,8,9,10,1::0,1,2,3,6,5,4,7,8,9,10,11::0,1,2,3,5,4,6,7,8,9,11,10::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11 --tensorboard=tensorboard/PossibleOrder/good --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=9845 --custom_alignment=0,1,2,3,4,5,7,6,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,9,6,7,8,5,10,11::0,11,2,3,4,5,6,7,8,9,10,1::0,1,2,3,6,5,4,7,8,9,10,11::0,1,2,3,5,4,6,7,8,9,11,10::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11 --tensorboard=tensorboard/PossibleOrder/good --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=3264 --custom_alignment=0,1,2,3,4,5,7,6,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,9,6,7,8,5,10,11::0,11,2,3,4,5,6,7,8,9,10,1::0,1,2,3,6,5,4,7,8,9,10,11::0,1,2,3,5,4,6,7,8,9,11,10::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11 --tensorboard=tensorboard/PossibleOrder/good --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=4859 --custom_alignment=0,1,2,3,4,5,7,6,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,9,6,7,8,5,10,11::0,11,2,3,4,5,6,7,8,9,10,1::0,1,2,3,6,5,4,7,8,9,10,11::0,1,2,3,5,4,6,7,8,9,11,10::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11 --tensorboard=tensorboard/PossibleOrder/good --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=9225 --custom_alignment=0,1,2,3,4,5,7,6,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,9,6,7,8,5,10,11::0,11,2,3,4,5,6,7,8,9,10,1::0,1,2,3,6,5,4,7,8,9,10,11::0,1,2,3,5,4,6,7,8,9,11,10::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11 --tensorboard=tensorboard/PossibleOrder/good --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# 1c5b2efc982ba716b99bb5765383024a
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=7891 --custom_alignment=0,1,2,3,4,5,7,6,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,9,6,7,8,5,10,11::0,11,2,3,4,5,6,7,8,9,10,1::0,1,2,3,6,5,4,7,8,9,10,11::0,1,2,3,5,4,6,7,8,9,11,10::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11 --tensorboard=tensorboard/PossibleOrder/good --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=4373 --custom_alignment=0,1,2,3,4,5,7,6,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,9,6,7,8,5,10,11::0,11,2,3,4,5,6,7,8,9,10,1::0,1,2,3,6,5,4,7,8,9,10,11::0,1,2,3,5,4,6,7,8,9,11,10::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11 --tensorboard=tensorboard/PossibleOrder/good --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=5874 --custom_alignment=0,1,2,3,4,5,7,6,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,9,6,7,8,5,10,11::0,11,2,3,4,5,6,7,8,9,10,1::0,1,2,3,6,5,4,7,8,9,10,11::0,1,2,3,5,4,6,7,8,9,11,10::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11 --tensorboard=tensorboard/PossibleOrder/good --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=6744 --custom_alignment=0,1,2,3,4,5,7,6,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,9,6,7,8,5,10,11::0,11,2,3,4,5,6,7,8,9,10,1::0,1,2,3,6,5,4,7,8,9,10,11::0,1,2,3,5,4,6,7,8,9,11,10::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11 --tensorboard=tensorboard/PossibleOrder/good --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=3468 --custom_alignment=0,1,2,3,4,5,7,6,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,9,6,7,8,5,10,11::0,11,2,3,4,5,6,7,8,9,10,1::0,1,2,3,6,5,4,7,8,9,10,11::0,1,2,3,5,4,6,7,8,9,11,10::0,1,2,3,4,5,6,7,8,9,10,11::0,1,2,3,4,5,6,7,8,9,10,11 --tensorboard=tensorboard/PossibleOrder/good --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# e4a367e29e01861a8bc15f53262cddaf
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=5265 --custom_alignment=5,2,3,4,11,0,9,8,7,6,1,10::6,1,10,2,7,5,11,0,3,4,9,8::2,4,8,9,5,6,1,0,7,10,11,3::1,9,0,5,11,2,8,6,3,7,4,10::3,7,2,9,8,1,11,0,6,10,5,4::0,6,4,2,5,3,11,9,7,10,1,8::1,2,11,7,9,6,8,10,4,5,3,0::9,7,10,1,0,6,5,2,3,11,8,4 --tensorboard=tensorboard/PossibleOrder/random --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# 1ff537e0739959c6529d7a25b72378b1
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=8121 --custom_alignment=11,9,7,8,6,4,2,1,10,5,0,3::8,6,3,4,11,9,1,5,7,0,10,2::0,1,4,8,5,11,3,6,10,9,2,7::7,8,6,5,0,3,9,11,10,2,1,4::1,5,4,8,7,2,3,9,10,6,11,0::1,11,7,4,6,0,5,9,10,3,2,8::6,11,1,10,7,0,5,9,3,4,8,2::2,4,3,7,1,11,5,10,0,8,9,6 --tensorboard=tensorboard/PossibleOrder/random --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# 8db9c920cb438574393eeafadd48ca14
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=3593 --custom_alignment=2,11,8,10,9,4,7,0,1,3,5,6::4,6,3,11,0,2,1,10,9,8,7,5::10,11,4,7,2,3,0,5,8,6,1,9::1,2,4,8,9,0,10,7,5,11,3,6::4,10,0,9,11,1,7,3,2,8,5,6::4,8,1,0,5,2,6,10,9,7,11,3::3,6,9,0,8,2,4,1,5,10,7,11::3,1,10,11,9,7,6,5,2,0,4,8 --tensorboard=tensorboard/PossibleOrder/random --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# d6d76c03d9be818c890e0ed4c700e77a
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=5310 --custom_alignment=2,10,6,5,4,7,8,9,3,0,11,1::0,9,4,8,10,1,5,6,2,11,7,3::0,6,1,3,10,8,7,11,9,5,2,4::0,8,9,5,3,4,2,11,10,1,6,7::7,8,11,6,4,1,3,0,2,10,9,5::1,7,5,3,9,8,4,2,11,6,0,10::6,4,8,10,1,2,5,9,3,0,7,11::1,3,8,7,5,10,6,2,11,9,4,0 --tensorboard=tensorboard/PossibleOrder/random --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# c3047fdfe8de529dcc7d71def5d686f0
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=8934 --custom_alignment=9,2,1,4,8,6,5,0,7,11,10,3::5,0,9,2,8,6,10,3,1,11,7,4::6,11,4,9,0,8,1,10,2,7,5,3::9,2,4,11,7,5,0,8,6,1,10,3::7,0,2,5,4,1,11,3,10,9,8,6::3,10,9,1,5,6,4,11,7,2,8,0::0,2,10,6,7,8,3,11,5,1,4,9::5,2,4,11,0,9,8,1,7,10,6,3 --tensorboard=tensorboard/PossibleOrder/random --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# 4e11ef008ee781869e167fedb4d13e70
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=7374 --custom_alignment=10,9,7,8,11,2,6,3,0,5,4,1::7,2,10,5,3,9,1,6,4,8,11,0::0,6,8,9,5,4,7,3,11,2,10,1::6,9,11,1,2,10,5,3,4,8,7,0::2,9,10,6,7,11,3,5,8,0,4,1::1,0,2,10,9,11,5,3,7,4,6,8::9,2,8,4,6,1,11,10,5,0,3,7::4,7,10,5,9,2,1,6,3,11,0,8 --tensorboard=tensorboard/PossibleOrder/random --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# 65824c682f1bc5e03f359159d888565c
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=8609 --custom_alignment=11,5,8,2,7,9,4,1,6,3,0,10::7,6,0,8,10,5,11,2,3,1,9,4::5,0,11,6,8,10,3,9,7,2,1,4::1,7,10,9,4,5,0,8,11,3,2,6::4,8,11,2,3,1,9,5,6,7,10,0::7,1,8,0,11,4,6,3,5,10,2,9::5,10,4,6,7,3,8,11,9,0,1,2::4,9,7,6,3,2,10,5,0,1,11,8 --tensorboard=tensorboard/PossibleOrder/random --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# 23475e80a05b5752016f9f55489a6d9e
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=2276 --custom_alignment=1,0,5,2,10,7,9,4,6,8,3,11::0,2,1,8,3,7,9,11,4,10,5,6::6,11,4,2,1,0,7,9,5,8,10,3::11,10,4,2,3,0,7,8,9,6,5,1::10,0,6,7,1,4,8,5,2,3,11,9::11,10,9,7,6,0,1,2,4,8,5,3::9,8,3,2,10,6,11,4,0,5,7,1::5,8,6,11,9,10,1,4,7,3,2,0 --tensorboard=tensorboard/PossibleOrder/random --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# f40ffd7cd237a6c548a3aecbd2fd2d21
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=1863 --custom_alignment=8,10,5,1,7,11,3,4,2,9,6,0::5,4,11,6,9,8,10,7,1,3,2,0::11,5,8,3,6,9,10,2,0,4,7,1::8,3,10,0,5,1,7,4,11,9,6,2::6,3,11,7,2,0,5,1,8,9,4,10::3,5,1,11,0,7,6,4,10,8,2,9::6,1,5,9,0,11,4,3,2,8,10,7::1,0,5,8,4,10,7,2,11,6,9,3 --tensorboard=tensorboard/PossibleOrder/random --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# f373da7884c585fc06a6c87ca48e4585
sbatch -J PossibleOrder submit.sh python 1.train.py --seed=7486 --custom_alignment=0,3,10,5,7,8,6,11,1,2,9,4::1,0,7,4,5,2,3,8,9,11,6,10::7,0,2,6,1,10,11,4,5,8,3,9::7,1,3,9,10,0,4,6,11,2,8,5::4,3,9,10,2,6,7,8,0,1,11,5::11,2,0,1,9,8,4,5,3,6,7,10::11,8,2,9,5,0,4,7,6,3,1,10::7,10,5,2,8,6,4,1,9,0,11,3 --tensorboard=tensorboard/PossibleOrder/random --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper  --custom_align_max_joints=12 --train_steps=1e7

# ========================================
# log
echo "================" >> ~/gpfs2/experiments.log
date >> ~/gpfs2/experiments.log
pwd >> ~/gpfs2/experiments.log
echo $0 >> ~/gpfs2/experiments.log
echo $exp_name >> ~/gpfs2/experiments.log
echo $description >> ~/gpfs2/experiments.log
squeue -O "JobID,Partition,Name,Nodelist,TimeUsed,UserName,StartTime,Schednodes,Command" --user=sliu1 -n $exp_name | head -n 10 >> ~/gpfs2/experiments.log
echo "================" >> ~/gpfs2/experiments.log


