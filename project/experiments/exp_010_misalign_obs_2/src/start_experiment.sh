#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

python generate.py

for seed in 0 1
do
    # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=300,400 --test_bodies=300,400 --seed=$seed --train_steps=1e6
    # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=300,400 --test_bodies=300,400 --seed=$seed --train_steps=1e6 --misalign_obs
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=300,400 --test_bodies=300,400 --seed=$seed --train_steps=1e6 --random_align_obs

    # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=400,500 --test_bodies=400,500 --seed=$seed --train_steps=1e6
    # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=400,500 --test_bodies=400,500 --seed=$seed --train_steps=1e6 --misalign_obs
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=400,500 --test_bodies=400,500 --seed=$seed --train_steps=1e6 --random_align_obs

    # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=500,600 --test_bodies=500,600 --seed=$seed --train_steps=1e6
    # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=500,600 --test_bodies=500,600 --seed=$seed --train_steps=1e6 --misalign_obs
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=500,600 --test_bodies=500,600 --seed=$seed --train_steps=1e6 --random_align_obs

    # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=300,400,500,600 --test_bodies=300,400,500,600 --seed=$seed --train_steps=1e6
    # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=300,400,500,600 --test_bodies=300,400,500,600 --seed=$seed --train_steps=1e6 --misalign_obs
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=300,400,500,600 --test_bodies=300,400,500,600 --seed=$seed --train_steps=1e6 --random_align_obs
done

