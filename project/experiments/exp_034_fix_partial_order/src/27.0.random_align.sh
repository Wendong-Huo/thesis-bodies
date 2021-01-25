#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)


for seed in $(seq 100 102)
do
    # Control refer to exp_026

    # Treatment

    # bodies with randomly added arms (and the alignment partially breaks)
    sbatch -J baseline9xx submit.sh python 1.train.py --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907 --train_steps=1e7 --seed=$seed 

    # randomly generated bodies together (and the alignment is an arbitrary one)
    sbatch -J baseline1xx submit.sh python 1.train.py --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --train_steps=1e7 --seed=$seed 

done