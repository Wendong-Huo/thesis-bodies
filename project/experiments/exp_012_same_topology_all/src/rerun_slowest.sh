#!/bin/sh
# kill them:
# 1786178             bluemoon            exp_012_same_topolognode419             1-05:34:03          sliu1               2021-01-08T19:00:50 (null)              /gpfs2/scratch/sliu1
# 1786179             bluemoon            exp_012_same_topolognode419             1-05:34:03          sliu1               2021-01-08T19:00:50 (null)              /gpfs2/scratch/sliu1
# 1786180             bluemoon            exp_012_same_topolognode419             1-05:34:03          sliu1               2021-01-08T19:00:50 (null)              /gpfs2/scratch/sliu1
# 1786181             bluemoon            exp_012_same_topolognode419             1-05:34:03          sliu1               2021-01-08T19:00:50 (null)              /gpfs2/scratch/sliu1
# 1786182             bluemoon            exp_012_same_topolognode419             1-05:34:03          sliu1               2021-01-08T19:00:50 (null)              /gpfs2/scratch/sliu1
# 1786151             bluemoon            exp_012_same_topolognode422             1-05:55:11          sliu1               2021-01-08T18:39:42 (null)              /gpfs2/scratch/sliu1
# 1786152             bluemoon            exp_012_same_topolognode422             1-05:55:11          sliu1               2021-01-08T18:39:42 (null)              /gpfs2/scratch/sliu1

# rerun:
rm output_data/tensorboard/model-500-501-502-504-505-506-508-509-510-511-512-513-514-515-517-518-general_only-sd14 -rf
sbatch -J rerun_slowest submit.sh python 1.train.py --seed=14 --train_bodies=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518 --test_bodies=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518 --realign_method=general_only

rm output_data/tensorboard/model-500-501-502-504-505-506-508-509-510-511-512-513-514-515-517-518-joints_only-sd14 -rf
sbatch -J rerun_slowest submit.sh python 1.train.py --seed=14 --train_bodies=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518 --test_bodies=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518 --realign_method=joints_only

rm output_data/tensorboard/model-500-501-502-504-505-506-508-509-510-511-512-513-514-515-517-518-feetcontact_only-sd14 -rf
sbatch -J rerun_slowest submit.sh python 1.train.py --seed=14 --train_bodies=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518 --test_bodies=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518 --realign_method=feetcontact_only

rm output_data/tensorboard/model-500-501-502-504-505-506-508-509-510-511-512-513-514-515-517-518-general_joints-sd14 -rf
sbatch -J rerun_slowest submit.sh python 1.train.py --seed=14 --train_bodies=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518 --test_bodies=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518 --realign_method=general_joints

rm output_data/tensorboard/model-500-501-502-504-505-506-508-509-510-511-512-513-514-515-517-518-general_feetcontact-sd14 -rf
sbatch -J rerun_slowest submit.sh python 1.train.py --seed=14 --train_bodies=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518 --test_bodies=500,501,502,504,505,506,508,509,510,511,512,513,514,515,517,518 --realign_method=general_feetcontact

rm output_data/tensorboard/model-500-501-502-504-505-506-507-508-509-511-512-513-514-515-516-518-joints_feetcontact-sd175 -rf
sbatch -J rerun_slowest submit.sh python 1.train.py --seed=175 --train_bodies=500,501,502,504,505,506,507,508,509,511,512,513,514,515,516,518 --test_bodies=500,501,502,504,505,506,507,508,509,511,512,513,514,515,516,518 --realign_method=joints_feetcontact

rm output_data/tensorboard/model-500-501-502-504-505-506-507-508-509-511-512-513-514-515-516-518-general_joints_feetcontact-sd175 -rf
sbatch -J rerun_slowest submit.sh python 1.train.py --seed=175 --train_bodies=500,501,502,504,505,506,507,508,509,511,512,513,514,515,516,518 --test_bodies=500,501,502,504,505,506,507,508,509,511,512,513,514,515,516,518 --realign_method=general_joints_feetcontact

