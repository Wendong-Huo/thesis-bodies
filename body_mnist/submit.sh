#!/bin/bash

#SBATCH --partition bluemoon
#SBATCH --mem 4G
#SBATCH -c 1
#SBATCH --output slurm.out/%j

cd ${SLURM_SUBMIT_DIR}

source activate thesis-bodies

echo $@
$@
