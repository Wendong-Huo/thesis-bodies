#!/bin/bash
#SBATCH --partition bluemoon
#SBATCH --mem 4G
#SBATCH -c 1
cd ${SLURM_SUBMIT_DIR}
source activate what_ever_your_conda_environment_is
echo $@
$@