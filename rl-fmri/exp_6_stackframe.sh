#!/bin/sh

# control:
# no frame stack
# treatment:
# stack 4 frames for observation (30,) -> (120,)

rm exp6/tb -r

# control
sbatch -J stackframe submit.sh python train_hard.py --exp=exp6 --train-bodies=400,500,600 --test-bodies=400,500,600 --stack_frames=1 --train-steps=3e6 --seed=0
sbatch -J stackframe submit.sh python train_hard.py --exp=exp6 --train-bodies=400,500,600 --test-bodies=400,500,600 --stack_frames=1 --train-steps=3e6 --seed=1
sbatch -J stackframe submit.sh python train_hard.py --exp=exp6 --train-bodies=400,500,600 --test-bodies=400,500,600 --stack_frames=1 --train-steps=3e6 --seed=2
sbatch -J stackframe submit.sh python train_hard.py --exp=exp6 --train-bodies=400,500,600 --test-bodies=400,500,600 --stack_frames=1 --train-steps=3e6 --seed=3
sbatch -J stackframe submit.sh python train_hard.py --exp=exp6 --train-bodies=400,500,600 --test-bodies=400,500,600 --stack_frames=1 --train-steps=3e6 --seed=4

# treatment
sbatch -J stackframe submit.sh python train_hard.py --exp=exp6 --train-bodies=400,500,600 --test-bodies=400,500,600 --stack_frames=4 --train-steps=3e6 --seed=0
sbatch -J stackframe submit.sh python train_hard.py --exp=exp6 --train-bodies=400,500,600 --test-bodies=400,500,600 --stack_frames=4 --train-steps=3e6 --seed=1
sbatch -J stackframe submit.sh python train_hard.py --exp=exp6 --train-bodies=400,500,600 --test-bodies=400,500,600 --stack_frames=4 --train-steps=3e6 --seed=2
sbatch -J stackframe submit.sh python train_hard.py --exp=exp6 --train-bodies=400,500,600 --test-bodies=400,500,600 --stack_frames=4 --train-steps=3e6 --seed=3
sbatch -J stackframe submit.sh python train_hard.py --exp=exp6 --train-bodies=400,500,600 --test-bodies=400,500,600 --stack_frames=4 --train-steps=3e6 --seed=4
