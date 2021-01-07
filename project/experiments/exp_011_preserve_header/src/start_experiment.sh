#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

for seed in 0 1 2 3 4 5 6 7 8 9
do
    # Control
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315 --test_bodies=300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415 --test_bodies=400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --test_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615 --test_bodies=600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615
    # Treatment
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315 --test_bodies=300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315 --random_align_obs
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415 --test_bodies=400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415 --random_align_obs
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --test_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --random_align_obs
    sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615 --test_bodies=600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615 --random_align_obs
done

