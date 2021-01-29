#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="802.1.v4_random_alignments"

description="With Arbitrary two alignments, do 100 runs for each and compare the log space difference."


for seed in $(seq 0 99)
do
# maybe_best_alignment in vanilla4:
    maybe_best_alignment=4,3,2,5,0,6,1,7::4,5,6,2,3,1,7,0::0,5,1,3,2,7,6,4::4,0,2,1,7,5,3,6
# 25aca1bfcda9ab49da1adcf14a804d7f

# maybe_worst_alignment in vanilla4:
    maybe_worst_alignment=6,2,3,0,4,5,7,1::2,1,3,5,6,7,4,0::7,4,0,3,1,2,6,5::6,4,2,5,7,3,0,1
# 31e3558068aa1c4101069f4373b7eafd

    sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/best  --custom_alignment=$maybe_best_alignment --seed=$seed --train_bodies=399,499,599,699 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8
    sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/worst --custom_alignment=$maybe_worst_alignment --seed=$seed --train_bodies=399,499,599,699 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8
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
