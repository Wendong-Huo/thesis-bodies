#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

# Job 0 Start
# 0 [317] [610]

            for case_id in 1 2 3 4 5 6
            do        
                sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=993 --train_bodies=317,610 --test_bodies=317,610 --topologies=diff --wrapper_type=Walker2DHopperCase$case_id
            done
        
# Job 0 End

# Job 1 Start
# 1 [304] [602]

            for case_id in 1 2 3 4 5 6
            do        
                sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=859 --train_bodies=304,602 --test_bodies=304,602 --topologies=diff --wrapper_type=Walker2DHopperCase$case_id
            done
        
# Job 1 End

# Job 2 Start
# 2 [301, 303, 304, 305, 306, 313, 314, 319] [600, 603, 605, 606, 608, 609, 618, 619]

            for case_id in 1 2 3 4 5 6
            do        
                sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=298 --train_bodies=301,303,304,305,306,313,314,319,600,603,605,606,608,609,618,619 --test_bodies=301,303,304,305,306,313,314,319,600,603,605,606,608,609,618,619 --topologies=diff --wrapper_type=Walker2DHopperCase$case_id
            done
        
# Job 2 End

# Job 3 Start
# 3 [301, 303, 304, 307, 309, 311, 315, 319] [600, 608, 612, 613, 614, 615, 617, 619]

            for case_id in 1 2 3 4 5 6
            do        
                sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=553 --train_bodies=301,303,304,307,309,311,315,319,600,608,612,613,614,615,617,619 --test_bodies=301,303,304,307,309,311,315,319,600,608,612,613,614,615,617,619 --topologies=diff --wrapper_type=Walker2DHopperCase$case_id
            done
        
# Job 3 End

