#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=0 --tensorboard=tensorboard_arms_help --train_bodies=998 --test_bodies=998
sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=1 --tensorboard=tensorboard_arms_help --train_bodies=998 --test_bodies=998
sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=2 --tensorboard=tensorboard_arms_help --train_bodies=998 --test_bodies=998

sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=0 --tensorboard=tensorboard_arms_help --train_bodies=997 --test_bodies=997
sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=1 --tensorboard=tensorboard_arms_help --train_bodies=997 --test_bodies=997
sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=2 --tensorboard=tensorboard_arms_help --train_bodies=997 --test_bodies=997
