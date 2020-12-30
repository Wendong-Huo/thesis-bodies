#!/bin/sh

# test different threshold
# 0.0 is equivalent to default ReLU
# see if lift the threshold can help multi-body policy (make things sparser)

rm exp4/tb -r

sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.0 --seed=0
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.0 --seed=1
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.0 --seed=2
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.0 --seed=3
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.0 --seed=4

sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.1 --seed=0
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.1 --seed=1
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.1 --seed=2
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.1 --seed=3
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.1 --seed=4

sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.2 --seed=0
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.2 --seed=1
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.2 --seed=2
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.2 --seed=3
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.2 --seed=4

sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.4 --seed=0
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.4 --seed=1
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.4 --seed=2
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.4 --seed=3
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.4 --seed=4

sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.8 --seed=0
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.8 --seed=1
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.8 --seed=2
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.8 --seed=3
sbatch -J threshold submit.sh python train_hard.py --exp=exp4 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=15 --train-steps=3e6 --threshold_threshold=0.8 --seed=4