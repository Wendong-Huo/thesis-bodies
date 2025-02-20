#!/bin/bash

#SBATCH --partition bluemoon
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output slurm.out/%j

cd ${SLURM_SUBMIT_DIR}

source activate thesis-bodies

time python train.py -n 20000000 --algo=ppo -tb experiment_results/tb_3xg20b_nobodyinfo/$1_$2 -f experiment_results/logs_3xg20b_nobodyinfo/$1_$2 --seed $2 --single-group $1


