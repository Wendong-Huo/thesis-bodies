#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit.sh
# ========================================

exp_name="801.2.train_variations"

description="For each type, 
we train the 100 robots one by one; repeat for 3 seeds;
We are going to pick 16 of those with highest learnability (measured at 2e6 steps);
This is to make sure we have valid test-beds (robots) in our experiments.
"

# This is not used. We actually select body by training using stack frame 4 and by shortest training time to pass learnability of 1500.

for seed in $(seq 0 2)
do
    for body in $(seq 300 349) $(seq 400 449) $(seq 500 549) $(seq 600 649)
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
