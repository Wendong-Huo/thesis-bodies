#!/bin/bash

#SBATCH --partition bluemoon
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output slurm.out/%j

cd ${SLURM_SUBMIT_DIR}

source activate thesis-bodies

time python train.py -n 20000000 --algo=ppo1 -tb experiment_results/tb_3x20ub_bodyinfo/$1_$2 -f experiment_results/logs_3x20ub_bodyinfo/$1_$2 --seed $2 --single-group $1


