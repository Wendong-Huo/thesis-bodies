#!/bin/bash

#SBATCH --partition short
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output slurm.out/%j

cd ${SLURM_SUBMIT_DIR}

source activate thesis-bodies

time python train.py --algo=ppo1 -tb tb_3x100_bodyinfo/$1_$2 -f logs_3x100_bodyinfo/$1_$2 --seed $2 --single-idx $1


