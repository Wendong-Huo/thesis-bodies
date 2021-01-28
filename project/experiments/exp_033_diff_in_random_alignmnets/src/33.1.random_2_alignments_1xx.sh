#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="TwoAlignments1xx"

description="We discovered difference in two random alignments for V4 bodies. Now let's test 1xx."


for seed in $(seq 0 99)
do

    alignment_1=5,2,1,3,0,4::1,3,4,0,2,5::3,5,1,2,4,0::5,2,3,4,1,0::2,4,3,5,1,0::3,1,4,2,0,5::0,1,2,4,5,3::5,2,3,4,1,0
    # 3368cd9a3e32b63ea6fc77bf96133aea
    alignment_2=3,1,4,5,2,0::1,3,0,5,4,2::3,2,1,5,0,4::0,5,3,2,1,4::5,2,3,4,1,0::0,1,4,2,5,3::2,4,5,1,3,0::1,5,0,3,4,2
    # cc0ecb88d504524f7c06a2584ff99c91

    sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/one --custom_alignment=$alignment_1 --seed=$seed --train_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=6
    sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/two --custom_alignment=$alignment_2 --seed=$seed --train_bodies=100,101,102,103,104,105,106,107 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=6
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
