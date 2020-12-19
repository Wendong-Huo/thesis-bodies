#!/bin/bash

#SBATCH --partition short
#SBATCH --mem 6G
#SBATCH -c 1
#SBATCH --output slurm.out/%j

cd ${SLURM_SUBMIT_DIR}

source activate thesis-bodies

python test_simple.py $1