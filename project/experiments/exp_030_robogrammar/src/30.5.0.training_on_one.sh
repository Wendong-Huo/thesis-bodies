#!/bin/sh
source activate thesis-bodies
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

exp_name="RoboGrammar"
description="Train on one, using VecNormal, see if body 1011-1020 can be trained."
# ========================================================================================

for body in $(seq 1011 1020)
do
    sbatch -J $exp_name submit.sh python 30.1.train.py --vec_normal --dataset_folder output_data/bodies --tensorboard tensorboard_vnorm --robo_bodies=$body
done

# ========================================================================================
# log
date >> ~/gpfs2/experiments.log
pwd >> ~/gpfs2/experiments.log
echo $0 >> ~/gpfs2/experiments.log
echo $exp_name >> ~/gpfs2/experiments.log
echo $description >> ~/gpfs2/experiments.log
