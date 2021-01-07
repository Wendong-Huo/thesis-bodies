#!/bin/sh
# part of the experiments didn't finish but time out, so need to rerun them.
source activate thesis-bodies

set -x


sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=4 --train_bodies=300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315 --test_bodies=300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315
sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=6 --train_bodies=300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315 --test_bodies=300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315 --random_align_obs
sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=0 --train_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --test_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515
sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=4 --train_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --test_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515
sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=6 --train_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --test_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515
sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=0 --train_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --test_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --random_align_obs
sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=3 --train_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --test_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --random_align_obs
sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=5 --train_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --test_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --random_align_obs
sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=6 --train_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --test_bodies=500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515 --random_align_obs
sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=4 --train_bodies=600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615 --test_bodies=600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615
sbatch -J exp_010_rerun submit.sh python 1.train.py --seed=6 --train_bodies=600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615 --test_bodies=600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615

