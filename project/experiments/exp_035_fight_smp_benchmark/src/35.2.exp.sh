#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-gpu.sh
# ========================================

exp_name="SMP_bodies"

description="Compare aligned and misaligned performance in SMP cheetahs"


for seed in 0 1 2
do
    sbatch -J $exp_name $submit_to python 35.1.fight_smp_benchmark.py --seed=$seed -f=$exp_name/cheetah/aligned --smp_bodies_aligned --train_steps=3e6 
    sbatch -J $exp_name $submit_to python 35.1.fight_smp_benchmark.py --seed=$seed -f=$exp_name/cheetah/misaligned --train_steps=3e6 
done

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

4-show-experiment-log.sh
