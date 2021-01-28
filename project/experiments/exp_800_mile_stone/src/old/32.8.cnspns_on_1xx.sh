#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit.sh
# ========================================

exp_name="CNSPNS1xx"

description="We want to see if cnspns can successfully find the right solution for 1xx. (which is unknown) This can additionally show CNSPNS architecture can find valid solutions."


for seed in 0 1 2
do

    sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name --train_bodies=100,101,102,103,104,105,106,107 --cnspns --seed=$seed --train_steps=1e7

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
