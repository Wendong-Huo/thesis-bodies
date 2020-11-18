#!/bin/bash

#SBATCH --partition short
#SBATCH --mem 1G
#SBATCH -c 1
#SBATCH --output slurm.out/%j

cd ${SLURM_SUBMIT_DIR}

source /users/s/l/sliu1/miniconda3/bin/activate thesis-bodies

# python cross_test.py --train $1
# python cross_test_group.py --train $1 --bodyinfo
python cross_test.py --algo=ppo --train=$1 --train_on=g10 --bodyinfo=_nobodyinfo
# python cross_test.py --algo=ppo1 --train=$1 --train_on=g10


