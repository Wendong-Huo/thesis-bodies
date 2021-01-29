#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
# ========================================

exp_name="PermutationFromGradientVanilla4"

description="Use pns and pns_init, so we can have gradient for pns weights, we regulate the weight, align them to permutation matrix when needed."


sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=68268 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=43567 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=42613 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=45891 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=21243 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=95939 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=97639 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=41993 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=86293 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=55026 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=80471 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=80966 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=48600 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=39512 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=52620 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=80186 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=17089 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=32230 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=18983 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4
sbatch -J PermutationFromGradientVanilla4 submit.sh python 1.train.py --seed=89688 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --pns --pns_init --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_steps=5e6 --tensorboard=tensorboard/PermutationFromGradientVanilla4

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


