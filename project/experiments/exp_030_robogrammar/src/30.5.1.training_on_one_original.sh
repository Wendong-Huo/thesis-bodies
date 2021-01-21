#!/bin/sh
source activate thesis-bodies
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
# ========================================================================================

exp_name="RoboGrammar_original"

description="Train on one, use the bodies from the paper, they have about 10 joints."


for body in $(seq 1000 1003)
do
    sbatch -J $exp_name submit.sh python 30.1.train.py --dataset_folder output_data/bodies --tensorboard tensorboard_original --robo_bodies=$body
done









# ========================================================================================
# log
echo "================" >> ~/gpfs2/experiments.log
date >> ~/gpfs2/experiments.log
pwd >> ~/gpfs2/experiments.log
echo $0 >> ~/gpfs2/experiments.log
echo $exp_name >> ~/gpfs2/experiments.log
echo $description >> ~/gpfs2/experiments.log
squeue -O "JobID,Partition,Name,Nodelist,TimeUsed,UserName,StartTime,Schednodes,Command" --user=sliu1 -n $exp_name | head -n 10 >> ~/gpfs2/experiments.log
echo "================" >> ~/gpfs2/experiments.log
