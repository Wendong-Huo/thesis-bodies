#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="TrainOnOne"

description="We have implemented CNSPNS function, so now we can train on one body, and later we can add other bodies in. The basic intention is to get a policy that can work well on 399 first."

for body in 399 499 599 699
do
    sbatch -J $exp_name $submit_to python 1.train.py --train_bodies=$body --cnspns -f=$exp_name --seed=0
    sbatch -J $exp_name $submit_to python 1.train.py --train_bodies=$body --cnspns -f=$exp_name --seed=1
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
