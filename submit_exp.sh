#!/bin/bash

#SBATCH --partition short
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output slurm.out/%j

# cd ${SLURM_SUBMIT_DIR}

# conda activate thesis-bodies

time python train.py -tb tb/$1_$2_$3 -f logs/$1_$2_$3 --powercoeff $1 $2 $3


