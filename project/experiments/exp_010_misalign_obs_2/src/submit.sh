#!/bin/bash
#SBATCH --partition bluemoon
#SBATCH --mem 2G
#SBATCH -c 1
#SBATCH --cores-per-socket 7
#SBATCH --output /users/s/l/sliu1/gpfs2/slurm.out/%j-%x

cd ${SLURM_SUBMIT_DIR}
source activate thesis-bodies
echo "Job Start at:"
date
# collect some data to optimize slurm command
echo "Node information:"
lscpu
hostname
echo ""
echo ""
echo "Job Command:"
echo $@
echo ""
$@
echo ""
echo "Job End at:"
date
echo ""
