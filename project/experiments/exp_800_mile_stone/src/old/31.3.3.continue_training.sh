#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
# ========================================

exp_name="ContinueV4"

description="We picked a ok policy, but 399 doesn't work very well, so we do additional training on that."



sbatch -J $exp_name submit.sh python 1.train.py --tensorboard=tensorboard/$exp_name --train_bodies=399,399,399,399,499,599,699,699 --test_bodies=399,399,399,399,499,599,699,699 --pns --initialize_weights_from=output_data/models/model-399-499-599-699-CustomAlignWrapper-mdd41d8cd98f00b204e9800998ecf8427e-pns-pns_init-sd80966/best_model.zip --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=1e7 --seed=0
sbatch -J $exp_name submit.sh python 1.train.py --tensorboard=tensorboard/$exp_name --train_bodies=399,399,399,399,499,599,699,699 --test_bodies=399,399,399,399,499,599,699,699 --pns --initialize_weights_from=output_data/models/model-399-499-599-699-CustomAlignWrapper-mdd41d8cd98f00b204e9800998ecf8427e-pns-pns_init-sd80966/best_model.zip --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=1e7 --seed=1
sbatch -J $exp_name submit.sh python 1.train.py --tensorboard=tensorboard/$exp_name --train_bodies=399,399,399,399,499,599,699,699 --test_bodies=399,399,399,399,499,599,699,699 --pns --initialize_weights_from=output_data/models/model-399-499-599-699-CustomAlignWrapper-mdd41d8cd98f00b204e9800998ecf8427e-pns-pns_init-sd80966/best_model.zip --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=1e7 --learning_rate=1e-4 --seed=4


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


