#!/bin/bash

#SBATCH --partition bluemoon
#SBATCH --mem 24G
#SBATCH -c 1
#SBATCH --output /home/liusida/thesis-bodies/cleaned_version/out.slurm/%j

cd /home/liusida/thesis-bodies/cleaned_version

source activate thesis-bodies

python _train.py --exp-name=$1 --seed=$2 --dataset=dataset/walker2d_10-v0 --single
