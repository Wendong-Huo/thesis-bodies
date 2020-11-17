#!/bin/bash

#SBATCH --partition short
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output slurm.out/%j

cd ${SLURM_SUBMIT_DIR}

source activate thesis-bodies

time python train.py --algo=ppo -tb experiment_results/tb_3xg10_nobodyinfo/$1_$2 -f experiment_results/logs_3xg10_nobodyinfo/$1_$2 --seed $2 --single-group $1


