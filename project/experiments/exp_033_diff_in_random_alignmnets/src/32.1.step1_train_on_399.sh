#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="On399"

description="Step1. Train on 399 with no contraints on PNS"

sbatch -J $exp_name $submit_to python 1.train.py --train_bodies=399 --test_bodies=399 --pns --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --tensorboard=tensorboard/step1 --seed=0
sbatch -J $exp_name $submit_to python 1.train.py --train_bodies=399 --test_bodies=399 --pns --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --tensorboard=tensorboard/step1 --seed=1


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
