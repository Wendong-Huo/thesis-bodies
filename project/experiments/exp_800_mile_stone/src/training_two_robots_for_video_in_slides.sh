#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="4slides_500_900"

description="
Training two robots for the video in slide 1.
One is the Ant,
The other is the HalfCheetah.
"

sbatch -J $exp_name $submit_to python 1.train.py --seed=0 -f=$exp_name/4video/ --train_bodies=500,900 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8
sbatch -J $exp_name $submit_to python 1.train.py --seed=0 -f=$exp_name/4video/ --train_bodies=500,900 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --custom_alignment=0,1,2,3,4,5,6,7::7,6,5,4,3,2,1,0
# sbatch -J $exp_name $submit_to python 1.train.py --seed=0 -f=$exp_name/4video/ --train_bodies=500,600 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --custom_alignment=0,1,2,3,4,5,6,7::7,6,5,4,3,2,1,0

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

