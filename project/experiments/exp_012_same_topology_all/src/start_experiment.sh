#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

bodies_300=300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315
bodies_400=400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415
bodies_500=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515
bodies_600=600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
    do
        # Control, reuse exp_010 [aligned]
        # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies
        
        # Treatment
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
        # reuse exp_011
        # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
        # reuse exp_011
        # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
        # reuse exp_010 [-ra]
        # sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact

    done
done


