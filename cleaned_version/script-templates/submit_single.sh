#!/bin/bash

#SBATCH --partition $partition
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output $cwd/out.slurm/%j

cd $cwd

source activate thesis-bodies

_CMD_="python _train.py --exp-name=$1 --exp-idx=$2 --single --single-idx=$2 --seed=$3 $4 --dataset=$dataset --n-timesteps=$5"
echo $_CMD_
eval $_CMD_
