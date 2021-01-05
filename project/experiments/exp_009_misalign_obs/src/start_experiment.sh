#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

python generate.py

for seed in 0 1
do
    for body in $(eval echo "{300..349}")
    do
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6 --stack_frames=4
    done

    for body in $(eval echo "{400..449}")
    do
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6 --stack_frames=4
    done

    for body in $(eval echo "{500..549}")
    do
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6 --stack_frames=4
    done

    for body in $(eval echo "{600..649}")
    do
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --train_bodies=$body --test_bodies=$body --seed=$seed --train_steps=1e6 --stack_frames=4
    done
done

