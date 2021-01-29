#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
# ========================================

exp_name="PermutationFromGradient"

description="Use pns and pns_init, so we can have gradient for pns weights, we regulate the weight, align them to permutation matrix when needed."


sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=68268 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=43567 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=42613 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=45891 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=21243 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=95939 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=97639 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=41993 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=86293 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=55026 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=80471 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=80966 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=48600 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=39512 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=52620 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=80186 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=17089 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=32230 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=18983 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=89688 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=82457 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=93005 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=6921 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=38804 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=67699 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=70608 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=37619 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=7877 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=83966 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=1871 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=73135 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=2496 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=47954 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=24675 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=31921 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=99059 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=797 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=49811 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=68755 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=80782 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=90535 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=81857 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=52489 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=84665 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=41504 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=49866 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=84212 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=96766 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=11723 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient
sbatch -J PermutationFromGradient submit.sh python 1.train.py --seed=43890 --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --pns --pns_init --train_steps=1e7 --tensorboard=tensorboard/PermutationFromGradient

# ========================================
# log
echo "================" >> ~/gpfs2/experiments.log
date >> ~/gpfs2/experiments.log
pwd >> ~/gpfs2/experiments.log
echo $0 >> ~/gpfs2/experiments.log
echo $exp_name >> ~/gpfs2/experiments.log
echo $description >> ~/gpfs2/experiments.log
squeue -O "JobID,Partition,Name,Nodelist,TimeUsed,UserName,StartTime,Schednodes,Command" --user=sliu1 -n $exp_name | head -n 10 >> ~/gpfs2/experiments.log
echo "================" >> ~/gpfs2/experiments.log


