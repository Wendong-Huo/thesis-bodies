#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="KeepTrainingV4"

description="We've got a workable model for 399. Now let's add 499,599,699 in, and train them together."


for seed in 0 1 2
do
    # Keep Training, adding 499,599,699
    sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/cnspns --train_bodies=399,499,599,699 --cnspns --model_filename output_data/models/TrainOnOneRe/model-399-sd101/best_model.zip --seed=$seed
    # Keep Training, fix CNS part
    sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/fix_cns --train_bodies=399,499,599,699 --cnspns --cnspns_fix_cns --model_filename output_data/models/TrainOnOneRe/model-399-sd101/best_model.zip --seed=$seed
    # Training from scratch (this might work this time because we adjusted the clip_range and vec_normal) (double the time)
    sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/from_scratch --train_bodies=399,499,599,699 --cnspns --seed=$seed --train_steps=4e6

    # Default Control
    sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/control --train_bodies=399,499,599,699 --topology_wrapper=CustomAlignWrapper --seed=$seed --train_steps=4e6
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
