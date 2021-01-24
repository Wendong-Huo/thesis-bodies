#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="RedoV4Baseline"

description="I need to redo this experiment, because PPO clip range was set to 0.4 instead of 0.2, and now everything will use vec_normal."

for seed in 0 1 2 3 4
do
    for body in 399 499 599 699
    do
        sbatch -J $exp_name $submit_to python 1.train.py --train_bodies=$body -f=$exp_name --seed=$seed
        sbatch -J $exp_name $submit_to python 1.train.py --train_bodies=$body -f=$exp_name --seed=$seed
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
