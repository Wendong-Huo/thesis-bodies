# check tomorrow:
VACC: 
/users/s/l/sliu1/gpfs2/tmp/rl-baselines3-zoo
sbatch submit.sh python train.py -tb tb --env Walker2DBulletEnv-v0
Submitted batch job 1736208
No VecNormal, Wrappered. can be used for initialization.

VACC:
/users/s/l/sliu1/gpfs2/thesis-bodies/rl-fmri
(thesis-bodies) [sliu1@vacc-user1 rl-fmri]$ sbatch submit.sh python train_hard.py --exp=exp_3 --train-bodies=400 --test-bodies=400
Submitted batch job 1736190
(thesis-bodies) [sliu1@vacc-user1 rl-fmri]$ sbatch submit.sh python train_hard.py --exp=exp_3 --train-bodies=500 --test-bodies=500
Submitted batch job 1736191
(thesis-bodies) [sliu1@vacc-user1 rl-fmri]$ sbatch submit.sh python train_hard.py --exp=exp_3 --train-bodies=600 --test-bodies=600
Submitted batch job 1736192
(thesis-bodies) [sliu1@vacc-user1 rl-fmri]$ sbatch submit.sh python train_hard.py --exp=exp_3 --train-bodies=300 --test-bodies=300 --seed=1
Submitted batch job 1736194
(thesis-bodies) [sliu1@vacc-user1 rl-fmri]$ sbatch submit.sh python train_hard.py --exp=exp_3 --train-bodies=400 --test-bodies=400 --seed=1
Submitted batch job 1736195
(thesis-bodies) [sliu1@vacc-user1 rl-fmri]$ sbatch submit.sh python train_hard.py --exp=exp_3 --train-bodies=600 --test-bodies=600 --seed=1
Submitted batch job 1736196
(thesis-bodies) [sliu1@vacc-user1 rl-fmri]$ sbatch submit.sh python train_hard.py --exp=exp_3 --train-bodies=500 --test-bodies=500 --seed=1
Submitted batch job 1736197

train on theirown.

VACC:
[rl-fmri]$ python train_hard.py --exp=exp_3 --train-bodies=400,500,600 --st-bodies=400,500,600 --num-venvs=15 --seed=1 --train-steps=3e6
train together on three bodies.

