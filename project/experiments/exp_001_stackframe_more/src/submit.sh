#!/bin/bash
#SBATCH --partition bluemoon
#SBATCH --mem 2G
#SBATCH -c 1
#SBATCH --output /users/s/l/sliu1/slurm.out/%j-%x

cd ${SLURM_SUBMIT_DIR}
source activate thesis-bodies
echo $@
$@