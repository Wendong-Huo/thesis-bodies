#!/bin/bash

#SBATCH --partition short
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output slurm.out/%j

cd ${SLURM_SUBMIT_DIR}

source activate baselines3-zoo

time python train.py -tb=tb/same_body --save-freq=10000 --seed=$1


