#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="TestImpl"

description="Previously the cnspns is much worse than the control on one body. I implemented more switches, so the cnspns architecture should be able to at least equal to the control."


for seed in 0 1 2
do
    for body in 399,499,599,699
    do
    # Control:
    # Train on one, with no cnspns
        sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/nocnspns --train_bodies=$body --seed=$seed

        sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/control --train_bodies=$body --cnspns --seed=$seed
    # Treatment:
    # 1. Train on one, start with identity matrix (pns), and fix pns, which should equivalent to the control.
        sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/equ --train_bodies=$body --cnspns --cnspns_start_with_identity --cnspns_fix_pns_bodies=$body --seed=$seed
    # 2. Train on one, start with identity matrix (pns), and fix general info, should be slightly better than the control, since the pns module can be adjusted.
        sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/adj --train_bodies=$body --cnspns --cnspns_start_with_identity --cnspns_fix_general_info --seed=$seed
    # 3. Train on one, start with identity matrix (pns), and fix general info, should be slightly better than the control, enable relu, since the pns module can be adjusted and there is additional non-linearity.
        sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/adj_relu --train_bodies=$body --cnspns --cnspns_start_with_identity --cnspns_fix_general_info --cnspns_relu --seed=$seed

    # 4. Train on one, simply fix general info, enable relu.
        sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/fgi_relu --train_bodies=$body --cnspns --cnspns_fix_general_info --cnspns_relu --seed=$seed

    # 4. Train on one, simply fix general info.
        sbatch -J $exp_name $submit_to python 1.train.py -f=$exp_name/fgi --train_bodies=$body --cnspns --cnspns_fix_general_info --seed=$seed
    done
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
