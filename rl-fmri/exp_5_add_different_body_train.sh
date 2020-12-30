#!/bin/sh

# control:
# train on one body, test on the same body
# treatment:
# train on two bodies, test on the same body


rm exp5/tb -r

# control
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400 --test-bodies=400 --num-venvs=12 --train-steps=3e6 --seed=0
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400 --test-bodies=400 --num-venvs=12 --train-steps=3e6 --seed=1
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400 --test-bodies=400 --num-venvs=12 --train-steps=3e6 --seed=2
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400 --test-bodies=400 --num-venvs=12 --train-steps=3e6 --seed=3
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400 --test-bodies=400 --num-venvs=12 --train-steps=3e6 --seed=4

sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=500 --test-bodies=500 --num-venvs=12 --train-steps=3e6 --seed=0
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=500 --test-bodies=500 --num-venvs=12 --train-steps=3e6 --seed=1
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=500 --test-bodies=500 --num-venvs=12 --train-steps=3e6 --seed=2
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=500 --test-bodies=500 --num-venvs=12 --train-steps=3e6 --seed=3
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=500 --test-bodies=500 --num-venvs=12 --train-steps=3e6 --seed=4

sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=600 --test-bodies=600 --num-venvs=12 --train-steps=3e6 --seed=0
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=600 --test-bodies=600 --num-venvs=12 --train-steps=3e6 --seed=1
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=600 --test-bodies=600 --num-venvs=12 --train-steps=3e6 --seed=2
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=600 --test-bodies=600 --num-venvs=12 --train-steps=3e6 --seed=3
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=600 --test-bodies=600 --num-venvs=12 --train-steps=3e6 --seed=4

# treatment
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,500 --test-bodies=400,500 --num-venvs=12 --train-steps=3e6 --seed=0
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,500 --test-bodies=400,500 --num-venvs=12 --train-steps=3e6 --seed=1
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,500 --test-bodies=400,500 --num-venvs=12 --train-steps=3e6 --seed=2
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,500 --test-bodies=400,500 --num-venvs=12 --train-steps=3e6 --seed=3
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,500 --test-bodies=400,500 --num-venvs=12 --train-steps=3e6 --seed=4

sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,600 --test-bodies=400,600 --num-venvs=12 --train-steps=3e6 --seed=0
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,600 --test-bodies=400,600 --num-venvs=12 --train-steps=3e6 --seed=1
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,600 --test-bodies=400,600 --num-venvs=12 --train-steps=3e6 --seed=2
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,600 --test-bodies=400,600 --num-venvs=12 --train-steps=3e6 --seed=3
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,600 --test-bodies=400,600 --num-venvs=12 --train-steps=3e6 --seed=4

sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=500,600 --test-bodies=500,600 --num-venvs=12 --train-steps=3e6 --seed=0
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=500,600 --test-bodies=500,600 --num-venvs=12 --train-steps=3e6 --seed=1
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=500,600 --test-bodies=500,600 --num-venvs=12 --train-steps=3e6 --seed=2
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=500,600 --test-bodies=500,600 --num-venvs=12 --train-steps=3e6 --seed=3
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=500,600 --test-bodies=500,600 --num-venvs=12 --train-steps=3e6 --seed=4

sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=12 --train-steps=3e6 --seed=0
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=12 --train-steps=3e6 --seed=1
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=12 --train-steps=3e6 --seed=2
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=12 --train-steps=3e6 --seed=3
sbatch -J add_diff_body submit.sh python train_hard.py --exp=exp5 --train-bodies=400,500,600 --test-bodies=400,500,600 --num-venvs=12 --train-steps=3e6 --seed=4
