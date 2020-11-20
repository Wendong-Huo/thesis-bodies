#!/bin/bash

#SBATCH --partition $partition
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output $cwd/out.slurm/%j

cd $cwd

source activate thesis-bodies

_CMD_="python _train.py --exp-name=$1 --exp-idx=$2 --body-ids=$3 --seed=$4 $5 --dataset=$dataset --n-timesteps=$6"
echo $_CMD_
eval $_CMD_