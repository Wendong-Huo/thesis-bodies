#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
# ========================================

exp_name="PermutationFromGradientGPU"

description="Use pns and pns_init, so we can have gradient for pns weights, we regulate the weight, align them to permutation matrix when needed."


sbatch -J PermutationFromGradientGPU submit-gpu.sh python 1.train.py --seed=98539 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradientGPU
sbatch -J PermutationFromGradientGPU submit-gpu.sh python 1.train.py --seed=77708 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradientGPU
sbatch -J PermutationFromGradientGPU submit-gpu.sh python 1.train.py --seed=5192 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradientGPU
sbatch -J PermutationFromGradientGPU submit-gpu.sh python 1.train.py --seed=98047 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradientGPU
sbatch -J PermutationFromGradientGPU submit-gpu.sh python 1.train.py --seed=50057 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradientGPU

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


