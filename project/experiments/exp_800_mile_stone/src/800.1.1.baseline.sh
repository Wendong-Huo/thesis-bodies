#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="800.1.baseline"

description="To make a baseline, we train a policy on each of Walker2D, HalfCheetah, Ant, and Hopper, 
using RL (we used PPO implemented in stable-baselines3, 
we will keep the program and hyperparameters the same throughout this thesis). 
We repeat this for 10 different seeds.
"


for seed in $(seq 0 9)
do
    for body in 399 499 599 699
    do
        sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name --seed=$seed --train_bodies=$body
    done
done

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

4-show-experiment-log.sh
