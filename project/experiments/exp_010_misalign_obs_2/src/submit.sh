#!/bin/bash
#SBATCH --partition short
#SBATCH --mem 2G
#SBATCH -c 1
#SBATCH --output /users/s/l/sliu1/slurm.out/%j-%x

cd ${SLURM_SUBMIT_DIR}
source activate thesis-bodies
echo $@
$@