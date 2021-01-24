#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit.sh
# ========================================

exp_name="SmallChannelSizeRedo"

description="I am super curious about small channel size. if sensor channel=8, motor channel=4, can it eventually learn to walk? If yes, the info pass through the channel must be very useful, and that might help generalization when we freeze the CNS.\
 I need to redo this experiment, because PPO clip range was set to 0.4 instead of 0.2, and now everything will use vec_normal."

for body in 399 499 599 699
do
    sbatch -J $exp_name $submit_to python 1.train.py --train_bodies=$body --cnspns --cnspns_sensor_channel=8 --cnspns_motor_channel=4 -f=$exp_name --seed=200 --train_steps=2e7
    sbatch -J $exp_name $submit_to python 1.train.py --train_bodies=$body --cnspns --cnspns_sensor_channel=8 --cnspns_motor_channel=4 -f=$exp_name --seed=201 --train_steps=2e7
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
