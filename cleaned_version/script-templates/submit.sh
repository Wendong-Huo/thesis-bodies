#!/bin/bash

#SBATCH --partition $partition
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output $cwd/out.slurm/%j

cd $cwd

source activate thesis-bodies

python _train.py --exp-name=$1 --seed=$2 --dataset=$dataset --single