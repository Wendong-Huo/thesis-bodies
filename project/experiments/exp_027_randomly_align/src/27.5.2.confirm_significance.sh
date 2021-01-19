#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

# Check significance


for seed in $(seq 0 10)
do
# best_alignment in mutate 2:
    best_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,7,6,0::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7
# 4eff43cf706a9cacdde5a2c8b68ff941
# worst_alignment in mutate 2:
    worst_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,7,6,5::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7
# 12b3d68266e041fb3f7a912208c742d8
    sbatch -J search_9xx_check_significance submit-short.sh python 1.train.py --seed=$seed --custom_alignment=$best_alignment --tensorboard=tensorboard/9xx_mutate_confirm --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
    sbatch -J search_9xx_check_significance submit-short.sh python 1.train.py --seed=$seed --custom_alignment=$worst_alignment --tensorboard=tensorboard/9xx_mutate_confirm --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# best_alignment in mutate 4:
    best_alignment=1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7
# 36494bcd6bc1c3838b1806fdfe235a38
# worst_alignment in mutate 4:
    worst_alignment=0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7
# bc301f1d5cf455bf3f7b3576b491d68d
    sbatch -J search_9xx_check_significance submit-short.sh python 1.train.py --seed=$seed --custom_alignment=$best_alignment --tensorboard=tensorboard/9xx_mutate_confirm --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
    sbatch -J search_9xx_check_significance submit-short.sh python 1.train.py --seed=$seed --custom_alignment=$worst_alignment --tensorboard=tensorboard/9xx_mutate_confirm --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# best_alignment in mutate 8:
    best_alignment=4,1,3,2,5,0,7,6::1,2,0,3,4,5,6,7::1,2,3,5,4,0,6,7::1,2,0,4,5,3,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7
# 62fe719074fc13bd2543f7f2bb21173b
# worst_alignment in mutate 8:
    worst_alignment=2,1,0,3,4,5,6,7::6,5,0,3,4,2,1,7::1,2,3,0,4,5,6,7::1,2,3,5,4,0,6,7::5,3,2,4,1,6,0,7::5,0,2,3,4,1,6,7::2,0,1,3,4,5,6,7::3,2,0,1,4,5,6,7
# 56acd7d427a9f0756e9d6392b1e68d69
    sbatch -J search_9xx_check_significance submit-short.sh python 1.train.py --seed=$seed --custom_alignment=$best_alignment --tensorboard=tensorboard/9xx_mutate_confirm --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
    sbatch -J search_9xx_check_significance submit-short.sh python 1.train.py --seed=$seed --custom_alignment=$worst_alignment --tensorboard=tensorboard/9xx_mutate_confirm --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# best_alignment in mutate 16:
    best_alignment=0,1,2,3,4,5,6,7::6,1,0,2,4,5,3,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,0,4,3,1,5,7::1,6,2,3,4,5,0,7::2,0,1,6,4,7,5,3::2,3,0,1,4,5,6,7
# c3ef005cd7b44e114050f60aedcd3ac8
# worst_alignment in mutate 16:
    worst_alignment=7,1,2,3,4,5,6,0::6,2,1,3,4,0,5,7::1,0,3,2,4,5,6,7::1,3,6,2,4,0,5,7::1,2,3,4,7,6,5,0::4,3,2,5,1,0,6,7::5,0,6,3,4,2,1,7::2,3,0,1,4,5,6,7
# 58d804231b4fef8dff98232e1903f8e9
    sbatch -J search_9xx_check_significance submit-short.sh python 1.train.py --seed=$seed --custom_alignment=$best_alignment --tensorboard=tensorboard/9xx_mutate_confirm --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
    sbatch -J search_9xx_check_significance submit-short.sh python 1.train.py --seed=$seed --custom_alignment=$worst_alignment --tensorboard=tensorboard/9xx_mutate_confirm --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# best_alignment in mutate 32:
    best_alignment=7,1,2,3,4,5,6,0::1,2,0,4,7,5,6,3::3,6,1,5,4,0,2,7::1,6,3,5,4,7,2,0::6,2,0,4,5,1,3,7::0,7,2,5,3,4,1,6::2,0,7,3,4,5,1,6::2,3,0,4,6,5,1,7
# 010fe7ded85ea075dd045f067458bd4b
# worst_alignment in mutate 32:
    worst_alignment=0,7,6,5,3,2,4,1::5,2,3,4,1,0,6,7::3,1,2,0,4,6,5,7::7,4,1,2,0,5,6,3::7,2,3,5,0,6,4,1::1,4,2,6,3,5,0,7::4,0,1,3,2,5,6,7::2,4,0,1,3,5,6,7
# 0ef1b0953b9e880b3f88f5ba0e7efe15
    sbatch -J search_9xx_check_significance submit-short.sh python 1.train.py --seed=$seed --custom_alignment=$best_alignment --tensorboard=tensorboard/9xx_mutate_confirm --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
    sbatch -J search_9xx_check_significance submit-short.sh python 1.train.py --seed=$seed --custom_alignment=$worst_alignment --tensorboard=tensorboard/9xx_mutate_confirm --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

done
