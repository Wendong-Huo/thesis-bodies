#!/bin/bash

#SBATCH --partition short
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output slurm.out/%j

cd ${SLURM_SUBMIT_DIR}

source activate thesis-bodies

time python train.py -tb tb/$1_$2 -f logs/$1_$2 --seed $2 --single-idx $1


