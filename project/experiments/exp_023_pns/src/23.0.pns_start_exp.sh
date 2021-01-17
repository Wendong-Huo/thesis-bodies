#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

sbatch -J PNS submit.sh python 1.train.py --train_bodies=900 --test_bodies=900 --pns --num_venvs=8
sbatch -J PNS submit.sh python 1.train.py --train_bodies=900,901 --test_bodies=900,901 --pns --num_venvs=8
sbatch -J PNS submit.sh python 1.train.py --train_bodies=900,901,902,903 --test_bodies=900,901,902,903 --pns --num_venvs=8
sbatch -J PNS submit.sh python 1.train.py --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907 --pns --num_venvs=8