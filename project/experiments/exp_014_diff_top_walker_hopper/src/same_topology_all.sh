#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

# Job 0 Start
bodies_300=308,317
bodies_400=408,417
bodies_500=508,517
bodies_600=608,617

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=993
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 0 End

# Job 1 Start
bodies_300=307,313
bodies_400=407,413
bodies_500=507,513
bodies_600=607,613

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=859
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 1 End

# Job 2 Start
bodies_300=302,312
bodies_400=402,412
bodies_500=502,512
bodies_600=602,612

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=298
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 2 End

# Job 3 Start
bodies_300=301,308
bodies_400=401,408
bodies_500=501,508
bodies_600=601,608

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=553
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 3 End

# Job 4 Start
bodies_300=304,310
bodies_400=404,410
bodies_500=504,510
bodies_600=604,610

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=672
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 4 End

# Job 5 Start
bodies_300=301,309,313,316
bodies_400=401,409,413,416
bodies_500=501,509,513,516
bodies_600=601,609,613,616

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=971
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 5 End

# Job 6 Start
bodies_300=309,313,317,319
bodies_400=409,413,417,419
bodies_500=509,513,517,519
bodies_600=609,613,617,619

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=27
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 6 End

# Job 7 Start
bodies_300=301,305,312,316
bodies_400=401,405,412,416
bodies_500=501,505,512,516
bodies_600=601,605,612,616

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=231
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 7 End

# Job 8 Start
bodies_300=304,306,307,315
bodies_400=404,406,407,415
bodies_500=504,506,507,515
bodies_600=604,606,607,615

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=306
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 8 End

# Job 9 Start
bodies_300=302,308,310,311
bodies_400=402,408,410,411
bodies_500=502,508,510,511
bodies_600=602,608,610,611

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=706
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 9 End

# Job 10 Start
bodies_300=300,302,304,305,308,309,314,319
bodies_400=400,402,404,405,408,409,414,419
bodies_500=500,502,504,505,508,509,514,519
bodies_600=600,602,604,605,608,609,614,619

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=496
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 10 End

# Job 11 Start
bodies_300=302,303,306,309,311,314,315,317
bodies_400=402,403,406,409,411,414,415,417
bodies_500=502,503,506,509,511,514,515,517
bodies_600=602,603,606,609,611,614,615,617

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=558
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 11 End

# Job 12 Start
bodies_300=301,303,309,313,314,315,316,317
bodies_400=401,403,409,413,414,415,416,417
bodies_500=501,503,509,513,514,515,516,517
bodies_600=601,603,609,613,614,615,616,617

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=784
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 12 End

# Job 13 Start
bodies_300=300,305,307,310,312,313,314,318
bodies_400=400,405,407,410,412,413,414,418
bodies_500=500,505,507,510,512,513,514,518
bodies_600=600,605,607,610,612,613,614,618

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=239
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 13 End

# Job 14 Start
bodies_300=300,301,303,305,306,309,310,319
bodies_400=400,401,403,405,406,409,410,419
bodies_500=500,501,503,505,506,509,510,519
bodies_600=600,601,603,605,606,609,610,619

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=578
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 14 End

# Job 15 Start
bodies_300=300,301,303,304,306,307,308,309,310,311,312,314,315,316,317,318
bodies_400=400,401,403,404,406,407,408,409,410,411,412,414,415,416,417,418
bodies_500=500,501,503,504,506,507,508,509,510,511,512,514,515,516,517,518
bodies_600=600,601,603,604,606,607,608,609,610,611,612,614,615,616,617,618

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=55
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 15 End

# Job 16 Start
bodies_300=300,301,302,303,304,305,306,308,310,311,312,313,316,317,318,319
bodies_400=400,401,402,403,404,405,406,408,410,411,412,413,416,417,418,419
bodies_500=500,501,502,503,504,505,506,508,510,511,512,513,516,517,518,519
bodies_600=600,601,602,603,604,605,606,608,610,611,612,613,616,617,618,619

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=906
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 16 End

# Job 17 Start
bodies_300=300,301,302,304,305,306,307,308,309,311,312,313,314,315,316,318
bodies_400=400,401,402,404,405,406,407,408,409,411,412,413,414,415,416,418
bodies_500=500,501,502,504,505,506,507,508,509,511,512,513,514,515,516,518
bodies_600=600,601,602,604,605,606,607,608,609,611,612,613,614,615,616,618

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=175
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 17 End

# Job 18 Start
bodies_300=300,301,302,304,305,306,308,309,310,311,312,313,314,315,317,318
bodies_400=400,401,402,404,405,406,408,409,410,411,412,413,414,415,417,418
bodies_500=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518
bodies_600=600,601,602,604,605,606,608,609,610,611,612,613,614,615,617,618

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=14
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 18 End

# Job 19 Start
bodies_300=301,302,303,304,305,306,307,308,309,310,312,313,314,315,317,319
bodies_400=401,402,403,404,405,406,407,408,409,410,412,413,414,415,417,419
bodies_500=501,502,503,504,505,506,507,508,509,510,512,513,514,515,517,519
bodies_600=601,602,603,604,605,606,607,608,609,610,612,613,614,615,617,619

for bodies in $bodies_300 $bodies_400 $bodies_500 $bodies_600 
do
    seed=77
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies

    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=feetcontact_only
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=joints_feetcontact
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$bodies --test_bodies=$bodies --realign_method=general_joints_feetcontact
done
# Job 19 End

