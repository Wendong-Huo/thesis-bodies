#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=0 --tensorboard=tensorboard_arms_help --train_bodies=998 --test_bodies=998
sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=1 --tensorboard=tensorboard_arms_help --train_bodies=998 --test_bodies=998
sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=2 --tensorboard=tensorboard_arms_help --train_bodies=998 --test_bodies=998

sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=3 --tensorboard=tensorboard_arms_help --train_bodies=997 --test_bodies=997
sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=4 --tensorboard=tensorboard_arms_help --train_bodies=997 --test_bodies=997
sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=5 --tensorboard=tensorboard_arms_help --train_bodies=997 --test_bodies=997

sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=6 --tensorboard=tensorboard_arms_help --train_bodies=996 --test_bodies=996
sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=7 --tensorboard=tensorboard_arms_help --train_bodies=996 --test_bodies=996
sbatch -J arms_help submit-gpu.sh python 1.train.py --seed=8 --tensorboard=tensorboard_arms_help --train_bodies=996 --test_bodies=996
