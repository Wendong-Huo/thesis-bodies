#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

# 7f3cc9487c112aca22805ac23bc0c681
# 4b8520a7ae65397410eaec55a816047f
# most meaningful alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 1d8e72563daf848f1836d1effe995201
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=1440 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1d8e72563daf848f1836d1effe995201
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=455 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1d8e72563daf848f1836d1effe995201
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=6113 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1d8e72563daf848f1836d1effe995201
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=4031 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1d8e72563daf848f1836d1effe995201
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=6421 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,3,4,5,6,0,7::4,0,2,3,1,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 759c5ece3371e8e0183567b78ae24780
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=9899 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,3,4,5,6,0,7::4,0,2,3,1,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 759c5ece3371e8e0183567b78ae24780
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=929 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,3,4,5,6,0,7::4,0,2,3,1,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 759c5ece3371e8e0183567b78ae24780
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=6894 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,3,4,5,6,0,7::4,0,2,3,1,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 759c5ece3371e8e0183567b78ae24780
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=375 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,3,4,5,6,0,7::4,0,2,3,1,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 759c5ece3371e8e0183567b78ae24780
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3244 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,3,4,5,6,0,7::4,0,2,3,1,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 5,1,2,3,4,0,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,6,4,5,1,7

# 3b29b44e708411cf09c16f1ab596325a
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2813 --custom_alignment=5,1,2,3,4,0,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,6,4,5,1,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3b29b44e708411cf09c16f1ab596325a
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2573 --custom_alignment=5,1,2,3,4,0,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,6,4,5,1,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3b29b44e708411cf09c16f1ab596325a
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=792 --custom_alignment=5,1,2,3,4,0,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,6,4,5,1,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3b29b44e708411cf09c16f1ab596325a
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=4878 --custom_alignment=5,1,2,3,4,0,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,6,4,5,1,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3b29b44e708411cf09c16f1ab596325a
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=9455 --custom_alignment=5,1,2,3,4,0,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,6,4,5,1,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,0,3,2,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 1c860df27c2a42cd511bd5265251d6ad
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=8385 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,0,3,2,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1c860df27c2a42cd511bd5265251d6ad
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=694 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,0,3,2,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1c860df27c2a42cd511bd5265251d6ad
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=163 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,0,3,2,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1c860df27c2a42cd511bd5265251d6ad
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=1903 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,0,3,2,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1c860df27c2a42cd511bd5265251d6ad
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=5798 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,0,3,2,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,7,6,5::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 12b3d68266e041fb3f7a912208c742d8
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3176 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,7,6,5::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 12b3d68266e041fb3f7a912208c742d8
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=101 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,7,6,5::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 12b3d68266e041fb3f7a912208c742d8
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=27 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,7,6,5::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 12b3d68266e041fb3f7a912208c742d8
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=1219 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,7,6,5::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 12b3d68266e041fb3f7a912208c742d8
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=8697 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,7,6,5::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,7,6,0::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 4eff43cf706a9cacdde5a2c8b68ff941
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=7815 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,7,6,0::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4eff43cf706a9cacdde5a2c8b68ff941
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=8511 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,7,6,0::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4eff43cf706a9cacdde5a2c8b68ff941
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=685 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,7,6,0::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4eff43cf706a9cacdde5a2c8b68ff941
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=6406 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,7,6,0::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4eff43cf706a9cacdde5a2c8b68ff941
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3589 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,7,6,0::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,5,3,4,0,6,7::1,2,3,4,0,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 593a29e4c11a7f07d72911bcdd764c93
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2410 --custom_alignment=0,1,2,3,4,5,6,7::1,2,5,3,4,0,6,7::1,2,3,4,0,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 593a29e4c11a7f07d72911bcdd764c93
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=6264 --custom_alignment=0,1,2,3,4,5,6,7::1,2,5,3,4,0,6,7::1,2,3,4,0,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 593a29e4c11a7f07d72911bcdd764c93
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2841 --custom_alignment=0,1,2,3,4,5,6,7::1,2,5,3,4,0,6,7::1,2,3,4,0,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 593a29e4c11a7f07d72911bcdd764c93
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3263 --custom_alignment=0,1,2,3,4,5,6,7::1,2,5,3,4,0,6,7::1,2,3,4,0,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 593a29e4c11a7f07d72911bcdd764c93
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=5091 --custom_alignment=0,1,2,3,4,5,6,7::1,2,5,3,4,0,6,7::1,2,3,4,0,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7

# 6e717ebd05c29282a3b51f8be93c0413
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=4851 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6e717ebd05c29282a3b51f8be93c0413
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3588 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6e717ebd05c29282a3b51f8be93c0413
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=7940 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6e717ebd05c29282a3b51f8be93c0413
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=4080 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6e717ebd05c29282a3b51f8be93c0413
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=1214 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,6,4,5,3,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 36ebcae1c44c0a0b302d0a9e579dc341
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3353 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,6,4,5,3,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 36ebcae1c44c0a0b302d0a9e579dc341
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=9781 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,6,4,5,3,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 36ebcae1c44c0a0b302d0a9e579dc341
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=6030 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,6,4,5,3,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 36ebcae1c44c0a0b302d0a9e579dc341
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=4913 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,6,4,5,3,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 36ebcae1c44c0a0b302d0a9e579dc341
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=4689 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,6,4,5,3,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::0,2,3,4,5,6,1,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7

# 1022220fa45e59ee6bfe52b7b6b87ee7
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=9708 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::0,2,3,4,5,6,1,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1022220fa45e59ee6bfe52b7b6b87ee7
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=5421 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::0,2,3,4,5,6,1,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1022220fa45e59ee6bfe52b7b6b87ee7
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2889 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::0,2,3,4,5,6,1,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1022220fa45e59ee6bfe52b7b6b87ee7
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3141 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::0,2,3,4,5,6,1,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1022220fa45e59ee6bfe52b7b6b87ee7
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2217 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::0,2,3,4,5,6,1,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 11580a5edd66000afce8cd8d579c3d7c
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=6845 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 11580a5edd66000afce8cd8d579c3d7c
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2176 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 11580a5edd66000afce8cd8d579c3d7c
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=6316 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 11580a5edd66000afce8cd8d579c3d7c
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=5824 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 11580a5edd66000afce8cd8d579c3d7c
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=8505 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,7,0,1,4,5,6,3

# 371b452d64b5b89b58383f1f8dae1f39
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=9901 --custom_alignment=0,1,2,3,4,5,6,7::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,7,0,1,4,5,6,3 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 371b452d64b5b89b58383f1f8dae1f39
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=430 --custom_alignment=0,1,2,3,4,5,6,7::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,7,0,1,4,5,6,3 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 371b452d64b5b89b58383f1f8dae1f39
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=6908 --custom_alignment=0,1,2,3,4,5,6,7::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,7,0,1,4,5,6,3 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 371b452d64b5b89b58383f1f8dae1f39
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=9168 --custom_alignment=0,1,2,3,4,5,6,7::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,7,0,1,4,5,6,3 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 371b452d64b5b89b58383f1f8dae1f39
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=484 --custom_alignment=0,1,2,3,4,5,6,7::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,7,0,1,4,5,6,3 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,6,0,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 419736fb96589086263b452cf7c958c6
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2668 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,6,0,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 419736fb96589086263b452cf7c958c6
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=9377 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,6,0,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 419736fb96589086263b452cf7c958c6
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=40 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,6,0,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 419736fb96589086263b452cf7c958c6
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3830 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,6,0,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 419736fb96589086263b452cf7c958c6
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=8032 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,6,0,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::3,2,1,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 1aec15d4b2053bd4fd4f4cd6e5e4789f
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=7182 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::3,2,1,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1aec15d4b2053bd4fd4f4cd6e5e4789f
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=683 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::3,2,1,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1aec15d4b2053bd4fd4f4cd6e5e4789f
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=416 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::3,2,1,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1aec15d4b2053bd4fd4f4cd6e5e4789f
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2400 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::3,2,1,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 1aec15d4b2053bd4fd4f4cd6e5e4789f
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=5911 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::3,2,1,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,7,2,3,4,5,6,1::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,6,0,5,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 04dc658f5b28acb7804d9c996fd3ad37
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2816 --custom_alignment=0,7,2,3,4,5,6,1::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,6,0,5,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 04dc658f5b28acb7804d9c996fd3ad37
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=5954 --custom_alignment=0,7,2,3,4,5,6,1::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,6,0,5,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 04dc658f5b28acb7804d9c996fd3ad37
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=6540 --custom_alignment=0,7,2,3,4,5,6,1::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,6,0,5,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 04dc658f5b28acb7804d9c996fd3ad37
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=8107 --custom_alignment=0,7,2,3,4,5,6,1::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,6,0,5,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 04dc658f5b28acb7804d9c996fd3ad37
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=4234 --custom_alignment=0,7,2,3,4,5,6,1::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,6,0,5,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,5,4,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# fdd01d80aa1711a3b532e4c4d20638fe
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3379 --custom_alignment=0,1,2,3,5,4,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# fdd01d80aa1711a3b532e4c4d20638fe
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=1011 --custom_alignment=0,1,2,3,5,4,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# fdd01d80aa1711a3b532e4c4d20638fe
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3404 --custom_alignment=0,1,2,3,5,4,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# fdd01d80aa1711a3b532e4c4d20638fe
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3997 --custom_alignment=0,1,2,3,5,4,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# fdd01d80aa1711a3b532e4c4d20638fe
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2641 --custom_alignment=0,1,2,3,5,4,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,5,0,3,4,2,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,3,1,0,4,5,6,7::2,3,0,1,4,5,6,7

# b0a3d36336c1793028467a73fb0efca0
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3210 --custom_alignment=0,1,2,3,4,5,6,7::1,5,0,3,4,2,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,3,1,0,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b0a3d36336c1793028467a73fb0efca0
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=9747 --custom_alignment=0,1,2,3,4,5,6,7::1,5,0,3,4,2,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,3,1,0,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b0a3d36336c1793028467a73fb0efca0
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=5283 --custom_alignment=0,1,2,3,4,5,6,7::1,5,0,3,4,2,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,3,1,0,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b0a3d36336c1793028467a73fb0efca0
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=7137 --custom_alignment=0,1,2,3,4,5,6,7::1,5,0,3,4,2,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,3,1,0,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b0a3d36336c1793028467a73fb0efca0
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=676 --custom_alignment=0,1,2,3,4,5,6,7::1,5,0,3,4,2,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,3,1,0,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,6,0,3,4,5,2,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,7,6

# 09cdb93e456f8edca8dd291072cc288b
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=9100 --custom_alignment=0,1,2,3,4,5,6,7::1,6,0,3,4,5,2,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 09cdb93e456f8edca8dd291072cc288b
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2848 --custom_alignment=0,1,2,3,4,5,6,7::1,6,0,3,4,5,2,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 09cdb93e456f8edca8dd291072cc288b
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=4849 --custom_alignment=0,1,2,3,4,5,6,7::1,6,0,3,4,5,2,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 09cdb93e456f8edca8dd291072cc288b
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=1199 --custom_alignment=0,1,2,3,4,5,6,7::1,6,0,3,4,5,2,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 09cdb93e456f8edca8dd291072cc288b
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=1513 --custom_alignment=0,1,2,3,4,5,6,7::1,6,0,3,4,5,2,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::5,2,3,0,4,1,6,7::1,6,3,4,5,0,2,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# ce4e0e682224beb4c14baf3196358d87
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=1571 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::5,2,3,0,4,1,6,7::1,6,3,4,5,0,2,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ce4e0e682224beb4c14baf3196358d87
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=1822 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::5,2,3,0,4,1,6,7::1,6,3,4,5,0,2,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ce4e0e682224beb4c14baf3196358d87
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3110 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::5,2,3,0,4,1,6,7::1,6,3,4,5,0,2,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ce4e0e682224beb4c14baf3196358d87
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=8117 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::5,2,3,0,4,1,6,7::1,6,3,4,5,0,2,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ce4e0e682224beb4c14baf3196358d87
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2607 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::5,2,3,0,4,1,6,7::1,6,3,4,5,0,2,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 2 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::3,0,1,2,4,5,6,7::2,3,0,1,4,5,6,7

# 9b845e092f4848a9372a335bdf1e6449
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=2267 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::3,0,1,2,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9b845e092f4848a9372a335bdf1e6449
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=9930 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::3,0,1,2,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9b845e092f4848a9372a335bdf1e6449
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=3587 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::3,0,1,2,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9b845e092f4848a9372a335bdf1e6449
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=1518 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::3,0,1,2,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9b845e092f4848a9372a335bdf1e6449
sbatch -J search_9xx_mutate_2 submit.sh python 1.train.py --seed=8624 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::3,0,1,2,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_2 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,5,3,4,2,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,7,4,5,6,1

# c86b38a8a63cde695295d67369629e74
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9989 --custom_alignment=0,1,5,3,4,2,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,7,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c86b38a8a63cde695295d67369629e74
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9738 --custom_alignment=0,1,5,3,4,2,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,7,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c86b38a8a63cde695295d67369629e74
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3070 --custom_alignment=0,1,5,3,4,2,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,7,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c86b38a8a63cde695295d67369629e74
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=4942 --custom_alignment=0,1,5,3,4,2,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,7,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c86b38a8a63cde695295d67369629e74
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1291 --custom_alignment=0,1,5,3,4,2,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,7,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 36494bcd6bc1c3838b1806fdfe235a38
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=888 --custom_alignment=1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 36494bcd6bc1c3838b1806fdfe235a38
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6957 --custom_alignment=1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 36494bcd6bc1c3838b1806fdfe235a38
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=4846 --custom_alignment=1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 36494bcd6bc1c3838b1806fdfe235a38
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1091 --custom_alignment=1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 36494bcd6bc1c3838b1806fdfe235a38
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9482 --custom_alignment=1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,2,1,6,4,5,3,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7

# 697f95d626cb194f719b2dc8d73788c9
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1108 --custom_alignment=0,2,1,6,4,5,3,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 697f95d626cb194f719b2dc8d73788c9
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=4727 --custom_alignment=0,2,1,6,4,5,3,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 697f95d626cb194f719b2dc8d73788c9
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9 --custom_alignment=0,2,1,6,4,5,3,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 697f95d626cb194f719b2dc8d73788c9
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2425 --custom_alignment=0,2,1,6,4,5,3,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 697f95d626cb194f719b2dc8d73788c9
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1499 --custom_alignment=0,2,1,6,4,5,3,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::7,0,2,3,4,5,6,1::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 5,1,2,3,4,0,6,7::1,2,0,5,4,3,6,7::1,2,3,0,4,5,6,7::1,2,4,3,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 511929c90aff6f0d7c04c5c5620dfc34
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5196 --custom_alignment=5,1,2,3,4,0,6,7::1,2,0,5,4,3,6,7::1,2,3,0,4,5,6,7::1,2,4,3,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 511929c90aff6f0d7c04c5c5620dfc34
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3458 --custom_alignment=5,1,2,3,4,0,6,7::1,2,0,5,4,3,6,7::1,2,3,0,4,5,6,7::1,2,4,3,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 511929c90aff6f0d7c04c5c5620dfc34
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3936 --custom_alignment=5,1,2,3,4,0,6,7::1,2,0,5,4,3,6,7::1,2,3,0,4,5,6,7::1,2,4,3,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 511929c90aff6f0d7c04c5c5620dfc34
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5065 --custom_alignment=5,1,2,3,4,0,6,7::1,2,0,5,4,3,6,7::1,2,3,0,4,5,6,7::1,2,4,3,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 511929c90aff6f0d7c04c5c5620dfc34
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=8016 --custom_alignment=5,1,2,3,4,0,6,7::1,2,0,5,4,3,6,7::1,2,3,0,4,5,6,7::1,2,4,3,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::2,1,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,4,3,2,5,6,7::7,0,1,3,4,5,6,2::2,0,3,1,4,5,6,7

# 7a34c10ce8501fe16e1c2da46ab109b5
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=4793 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::2,1,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,4,3,2,5,6,7::7,0,1,3,4,5,6,2::2,0,3,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 7a34c10ce8501fe16e1c2da46ab109b5
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=7249 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::2,1,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,4,3,2,5,6,7::7,0,1,3,4,5,6,2::2,0,3,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 7a34c10ce8501fe16e1c2da46ab109b5
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=8167 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::2,1,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,4,3,2,5,6,7::7,0,1,3,4,5,6,2::2,0,3,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 7a34c10ce8501fe16e1c2da46ab109b5
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=778 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::2,1,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,4,3,2,5,6,7::7,0,1,3,4,5,6,2::2,0,3,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 7a34c10ce8501fe16e1c2da46ab109b5
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3821 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::2,1,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,4,3,2,5,6,7::7,0,1,3,4,5,6,2::2,0,3,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::0,2,1,3,4,5,6,7::4,3,0,1,2,5,6,7

# 8a88c210e863917704a401cfd1a00c7f
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5913 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::0,2,1,3,4,5,6,7::4,3,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8a88c210e863917704a401cfd1a00c7f
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2627 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::0,2,1,3,4,5,6,7::4,3,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8a88c210e863917704a401cfd1a00c7f
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=65 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::0,2,1,3,4,5,6,7::4,3,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8a88c210e863917704a401cfd1a00c7f
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2713 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::0,2,1,3,4,5,6,7::4,3,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8a88c210e863917704a401cfd1a00c7f
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1340 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::0,2,1,3,4,5,6,7::4,3,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,7,3,4,5,6,0::1,2,5,0,4,3,6,7::6,2,3,4,5,0,1,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# dd00be3078d6bc8c45c446dfc964d0b1
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3678 --custom_alignment=0,1,2,3,4,5,6,7::1,2,7,3,4,5,6,0::1,2,5,0,4,3,6,7::6,2,3,4,5,0,1,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# dd00be3078d6bc8c45c446dfc964d0b1
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2548 --custom_alignment=0,1,2,3,4,5,6,7::1,2,7,3,4,5,6,0::1,2,5,0,4,3,6,7::6,2,3,4,5,0,1,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# dd00be3078d6bc8c45c446dfc964d0b1
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=366 --custom_alignment=0,1,2,3,4,5,6,7::1,2,7,3,4,5,6,0::1,2,5,0,4,3,6,7::6,2,3,4,5,0,1,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# dd00be3078d6bc8c45c446dfc964d0b1
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6389 --custom_alignment=0,1,2,3,4,5,6,7::1,2,7,3,4,5,6,0::1,2,5,0,4,3,6,7::6,2,3,4,5,0,1,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# dd00be3078d6bc8c45c446dfc964d0b1
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5649 --custom_alignment=0,1,2,3,4,5,6,7::1,2,7,3,4,5,6,0::1,2,5,0,4,3,6,7::6,2,3,4,5,0,1,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::1,4,0,3,2,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,7,6::2,3,0,1,4,5,6,7

# 0a99551135d8fcbbc0e9660f723ec91d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3400 --custom_alignment=0,1,2,3,4,5,6,7::1,4,0,3,2,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,7,6::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0a99551135d8fcbbc0e9660f723ec91d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=7020 --custom_alignment=0,1,2,3,4,5,6,7::1,4,0,3,2,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,7,6::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0a99551135d8fcbbc0e9660f723ec91d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1879 --custom_alignment=0,1,2,3,4,5,6,7::1,4,0,3,2,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,7,6::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0a99551135d8fcbbc0e9660f723ec91d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=8883 --custom_alignment=0,1,2,3,4,5,6,7::1,4,0,3,2,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,7,6::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0a99551135d8fcbbc0e9660f723ec91d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1730 --custom_alignment=0,1,2,3,4,5,6,7::1,4,0,3,2,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,7,6::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::5,2,0,3,4,7,6,1::1,2,3,0,4,5,6,7::1,2,0,4,5,3,6,7::4,2,3,1,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 82bc2dfbbacbcdbe1d2491705b3b4c62
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1854 --custom_alignment=0,1,2,3,4,5,6,7::5,2,0,3,4,7,6,1::1,2,3,0,4,5,6,7::1,2,0,4,5,3,6,7::4,2,3,1,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 82bc2dfbbacbcdbe1d2491705b3b4c62
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1002 --custom_alignment=0,1,2,3,4,5,6,7::5,2,0,3,4,7,6,1::1,2,3,0,4,5,6,7::1,2,0,4,5,3,6,7::4,2,3,1,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 82bc2dfbbacbcdbe1d2491705b3b4c62
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=330 --custom_alignment=0,1,2,3,4,5,6,7::5,2,0,3,4,7,6,1::1,2,3,0,4,5,6,7::1,2,0,4,5,3,6,7::4,2,3,1,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 82bc2dfbbacbcdbe1d2491705b3b4c62
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2622 --custom_alignment=0,1,2,3,4,5,6,7::5,2,0,3,4,7,6,1::1,2,3,0,4,5,6,7::1,2,0,4,5,3,6,7::4,2,3,1,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 82bc2dfbbacbcdbe1d2491705b3b4c62
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=860 --custom_alignment=0,1,2,3,4,5,6,7::5,2,0,3,4,7,6,1::1,2,3,0,4,5,6,7::1,2,0,4,5,3,6,7::4,2,3,1,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,0,5,6,4,7::1,0,2,3,6,5,4,7::6,5,1,3,4,0,2,7::2,3,0,1,4,5,6,7

# 66e492e2dd0e8296002ee91c3aa871b0
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6805 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,0,5,6,4,7::1,0,2,3,6,5,4,7::6,5,1,3,4,0,2,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 66e492e2dd0e8296002ee91c3aa871b0
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6032 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,0,5,6,4,7::1,0,2,3,6,5,4,7::6,5,1,3,4,0,2,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 66e492e2dd0e8296002ee91c3aa871b0
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=228 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,0,5,6,4,7::1,0,2,3,6,5,4,7::6,5,1,3,4,0,2,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 66e492e2dd0e8296002ee91c3aa871b0
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=7168 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,0,5,6,4,7::1,0,2,3,6,5,4,7::6,5,1,3,4,0,2,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 66e492e2dd0e8296002ee91c3aa871b0
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9791 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,0,5,6,4,7::1,0,2,3,6,5,4,7::6,5,1,3,4,0,2,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,6,3,4,5,2,0,7::1,0,2,4,3,5,6,7::2,0,1,3,4,5,6,7::5,3,0,1,4,2,6,7

# f2a2be8c9bb72954370569ad04a2f136
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1645 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,6,3,4,5,2,0,7::1,0,2,4,3,5,6,7::2,0,1,3,4,5,6,7::5,3,0,1,4,2,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f2a2be8c9bb72954370569ad04a2f136
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3245 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,6,3,4,5,2,0,7::1,0,2,4,3,5,6,7::2,0,1,3,4,5,6,7::5,3,0,1,4,2,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f2a2be8c9bb72954370569ad04a2f136
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6390 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,6,3,4,5,2,0,7::1,0,2,4,3,5,6,7::2,0,1,3,4,5,6,7::5,3,0,1,4,2,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f2a2be8c9bb72954370569ad04a2f136
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=4008 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,6,3,4,5,2,0,7::1,0,2,4,3,5,6,7::2,0,1,3,4,5,6,7::5,3,0,1,4,2,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f2a2be8c9bb72954370569ad04a2f136
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5202 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,6,3,4,5,2,0,7::1,0,2,4,3,5,6,7::2,0,1,3,4,5,6,7::5,3,0,1,4,2,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7

# bc301f1d5cf455bf3f7b3576b491d68d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6544 --custom_alignment=0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# bc301f1d5cf455bf3f7b3576b491d68d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=7139 --custom_alignment=0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# bc301f1d5cf455bf3f7b3576b491d68d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2157 --custom_alignment=0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# bc301f1d5cf455bf3f7b3576b491d68d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6527 --custom_alignment=0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# bc301f1d5cf455bf3f7b3576b491d68d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=807 --custom_alignment=0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,0,6,7::1,2,3,4,5,6,0,7::1,6,2,3,4,5,0,7::4,0,1,3,2,5,6,7::2,3,0,1,4,5,6,7

# 6e4f9b11cc39e74f9a58ca5ec0c99396
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=8507 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,0,6,7::1,2,3,4,5,6,0,7::1,6,2,3,4,5,0,7::4,0,1,3,2,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6e4f9b11cc39e74f9a58ca5ec0c99396
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2114 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,0,6,7::1,2,3,4,5,6,0,7::1,6,2,3,4,5,0,7::4,0,1,3,2,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6e4f9b11cc39e74f9a58ca5ec0c99396
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3502 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,0,6,7::1,2,3,4,5,6,0,7::1,6,2,3,4,5,0,7::4,0,1,3,2,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6e4f9b11cc39e74f9a58ca5ec0c99396
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6008 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,0,6,7::1,2,3,4,5,6,0,7::1,6,2,3,4,5,0,7::4,0,1,3,2,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6e4f9b11cc39e74f9a58ca5ec0c99396
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=7107 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,3,2,4,5,0,6,7::1,2,3,4,5,6,0,7::1,6,2,3,4,5,0,7::4,0,1,3,2,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,4,3,2,1,5,6,7::1,2,0,3,4,5,6,7::1,2,4,0,6,5,3,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# f6969a095c5c3ba8e9807ce7b914186d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5007 --custom_alignment=0,4,3,2,1,5,6,7::1,2,0,3,4,5,6,7::1,2,4,0,6,5,3,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f6969a095c5c3ba8e9807ce7b914186d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5831 --custom_alignment=0,4,3,2,1,5,6,7::1,2,0,3,4,5,6,7::1,2,4,0,6,5,3,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f6969a095c5c3ba8e9807ce7b914186d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=959 --custom_alignment=0,4,3,2,1,5,6,7::1,2,0,3,4,5,6,7::1,2,4,0,6,5,3,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f6969a095c5c3ba8e9807ce7b914186d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=751 --custom_alignment=0,4,3,2,1,5,6,7::1,2,0,3,4,5,6,7::1,2,4,0,6,5,3,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f6969a095c5c3ba8e9807ce7b914186d
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3428 --custom_alignment=0,4,3,2,1,5,6,7::1,2,0,3,4,5,6,7::1,2,4,0,6,5,3,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,5,4,3,6,0,7::1,0,2,3,4,5,6,7::5,0,1,3,4,2,6,7::2,3,0,1,4,5,6,7

# 228112ca12795b83ce053ef1193cb365
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9785 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,5,4,3,6,0,7::1,0,2,3,4,5,6,7::5,0,1,3,4,2,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 228112ca12795b83ce053ef1193cb365
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5836 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,5,4,3,6,0,7::1,0,2,3,4,5,6,7::5,0,1,3,4,2,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 228112ca12795b83ce053ef1193cb365
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3270 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,5,4,3,6,0,7::1,0,2,3,4,5,6,7::5,0,1,3,4,2,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 228112ca12795b83ce053ef1193cb365
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=4672 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,5,4,3,6,0,7::1,0,2,3,4,5,6,7::5,0,1,3,4,2,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 228112ca12795b83ce053ef1193cb365
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9901 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,7,3,4,5,0,6,2::1,2,5,4,3,6,0,7::1,0,2,3,4,5,6,7::5,0,1,3,4,2,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 7,1,2,3,4,5,6,0::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,7,5,6,0,4::1,0,2,4,3,6,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 95abb3ef53a8840a4f4d2a8c7ea49824
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9959 --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,7,5,6,0,4::1,0,2,4,3,6,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 95abb3ef53a8840a4f4d2a8c7ea49824
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9933 --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,7,5,6,0,4::1,0,2,4,3,6,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 95abb3ef53a8840a4f4d2a8c7ea49824
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9445 --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,7,5,6,0,4::1,0,2,4,3,6,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 95abb3ef53a8840a4f4d2a8c7ea49824
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2935 --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,7,5,6,0,4::1,0,2,4,3,6,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 95abb3ef53a8840a4f4d2a8c7ea49824
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6416 --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,7,5,6,0,4::1,0,2,4,3,6,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::1,3,0,2,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7

# c8f3c21ae0fb971ba9044dcd9ff9f4c8
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=9708 --custom_alignment=0,1,2,3,4,5,6,7::1,3,0,2,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c8f3c21ae0fb971ba9044dcd9ff9f4c8
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=974 --custom_alignment=0,1,2,3,4,5,6,7::1,3,0,2,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c8f3c21ae0fb971ba9044dcd9ff9f4c8
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6505 --custom_alignment=0,1,2,3,4,5,6,7::1,3,0,2,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c8f3c21ae0fb971ba9044dcd9ff9f4c8
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=7648 --custom_alignment=0,1,2,3,4,5,6,7::1,3,0,2,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c8f3c21ae0fb971ba9044dcd9ff9f4c8
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=6046 --custom_alignment=0,1,2,3,4,5,6,7::1,3,0,2,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::0,1,2,3,4,5,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::5,2,0,3,4,1,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,7,5,6,4::2,6,0,1,4,5,7,3

# 60dd20d9cb187aa9c13d3ec8f07bd58f
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5673 --custom_alignment=0,1,2,3,4,5,6,7::5,2,0,3,4,1,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,7,5,6,4::2,6,0,1,4,5,7,3 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 60dd20d9cb187aa9c13d3ec8f07bd58f
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3574 --custom_alignment=0,1,2,3,4,5,6,7::5,2,0,3,4,1,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,7,5,6,4::2,6,0,1,4,5,7,3 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 60dd20d9cb187aa9c13d3ec8f07bd58f
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=4443 --custom_alignment=0,1,2,3,4,5,6,7::5,2,0,3,4,1,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,7,5,6,4::2,6,0,1,4,5,7,3 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 60dd20d9cb187aa9c13d3ec8f07bd58f
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2115 --custom_alignment=0,1,2,3,4,5,6,7::5,2,0,3,4,1,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,7,5,6,4::2,6,0,1,4,5,7,3 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 60dd20d9cb187aa9c13d3ec8f07bd58f
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=636 --custom_alignment=0,1,2,3,4,5,6,7::5,2,0,3,4,1,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,7,5,6,4::2,6,0,1,4,5,7,3 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 2,1,0,3,4,5,6,7::1,2,0,3,4,5,6,7::1,6,3,0,4,5,2,7::1,2,6,4,5,0,3,7::1,2,3,4,5,6,0,7::2,0,1,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 4e0c5a20c4da38435e0eb14a69dfa077
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5938 --custom_alignment=2,1,0,3,4,5,6,7::1,2,0,3,4,5,6,7::1,6,3,0,4,5,2,7::1,2,6,4,5,0,3,7::1,2,3,4,5,6,0,7::2,0,1,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4e0c5a20c4da38435e0eb14a69dfa077
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3479 --custom_alignment=2,1,0,3,4,5,6,7::1,2,0,3,4,5,6,7::1,6,3,0,4,5,2,7::1,2,6,4,5,0,3,7::1,2,3,4,5,6,0,7::2,0,1,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4e0c5a20c4da38435e0eb14a69dfa077
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2029 --custom_alignment=2,1,0,3,4,5,6,7::1,2,0,3,4,5,6,7::1,6,3,0,4,5,2,7::1,2,6,4,5,0,3,7::1,2,3,4,5,6,0,7::2,0,1,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4e0c5a20c4da38435e0eb14a69dfa077
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=3102 --custom_alignment=2,1,0,3,4,5,6,7::1,2,0,3,4,5,6,7::1,6,3,0,4,5,2,7::1,2,6,4,5,0,3,7::1,2,3,4,5,6,0,7::2,0,1,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4e0c5a20c4da38435e0eb14a69dfa077
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=2287 --custom_alignment=2,1,0,3,4,5,6,7::1,2,0,3,4,5,6,7::1,6,3,0,4,5,2,7::1,2,6,4,5,0,3,7::1,2,3,4,5,6,0,7::2,0,1,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 4 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::6,2,3,0,4,5,1,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::3,0,7,1,4,5,6,2::2,0,4,3,1,5,6,7::2,3,0,1,4,5,6,7

# ee3a744ab7559b9f2a17bc2debb4540b
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=5858 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::6,2,3,0,4,5,1,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::3,0,7,1,4,5,6,2::2,0,4,3,1,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ee3a744ab7559b9f2a17bc2debb4540b
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=4822 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::6,2,3,0,4,5,1,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::3,0,7,1,4,5,6,2::2,0,4,3,1,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ee3a744ab7559b9f2a17bc2debb4540b
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=666 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::6,2,3,0,4,5,1,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::3,0,7,1,4,5,6,2::2,0,4,3,1,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ee3a744ab7559b9f2a17bc2debb4540b
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=1069 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::6,2,3,0,4,5,1,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::3,0,7,1,4,5,6,2::2,0,4,3,1,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ee3a744ab7559b9f2a17bc2debb4540b
sbatch -J search_9xx_mutate_4 submit.sh python 1.train.py --seed=4903 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::6,2,3,0,4,5,1,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::3,0,7,1,4,5,6,2::2,0,4,3,1,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_4 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 6,1,2,3,4,5,0,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::5,3,2,4,1,0,7,6::1,4,3,2,5,6,0,7::1,0,2,3,7,5,6,4::2,0,1,3,4,5,6,7::2,3,5,1,4,0,7,6

# ba427c0a54bcf5d91e9de64dbde64586
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=9206 --custom_alignment=6,1,2,3,4,5,0,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::5,3,2,4,1,0,7,6::1,4,3,2,5,6,0,7::1,0,2,3,7,5,6,4::2,0,1,3,4,5,6,7::2,3,5,1,4,0,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ba427c0a54bcf5d91e9de64dbde64586
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=1927 --custom_alignment=6,1,2,3,4,5,0,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::5,3,2,4,1,0,7,6::1,4,3,2,5,6,0,7::1,0,2,3,7,5,6,4::2,0,1,3,4,5,6,7::2,3,5,1,4,0,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ba427c0a54bcf5d91e9de64dbde64586
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2388 --custom_alignment=6,1,2,3,4,5,0,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::5,3,2,4,1,0,7,6::1,4,3,2,5,6,0,7::1,0,2,3,7,5,6,4::2,0,1,3,4,5,6,7::2,3,5,1,4,0,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ba427c0a54bcf5d91e9de64dbde64586
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6073 --custom_alignment=6,1,2,3,4,5,0,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::5,3,2,4,1,0,7,6::1,4,3,2,5,6,0,7::1,0,2,3,7,5,6,4::2,0,1,3,4,5,6,7::2,3,5,1,4,0,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ba427c0a54bcf5d91e9de64dbde64586
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=9641 --custom_alignment=6,1,2,3,4,5,0,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::5,3,2,4,1,0,7,6::1,4,3,2,5,6,0,7::1,0,2,3,7,5,6,4::2,0,1,3,4,5,6,7::2,3,5,1,4,0,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,1,2,3,5,4,6,7::1,2,4,3,0,5,6,7::1,2,3,0,4,5,6,7::2,1,3,4,5,6,0,7::1,2,3,4,7,6,0,5::7,0,4,3,2,5,6,1::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7

# f5342d1fdf15744c668c0a01ae75cffc
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7039 --custom_alignment=0,1,2,3,5,4,6,7::1,2,4,3,0,5,6,7::1,2,3,0,4,5,6,7::2,1,3,4,5,6,0,7::1,2,3,4,7,6,0,5::7,0,4,3,2,5,6,1::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f5342d1fdf15744c668c0a01ae75cffc
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7234 --custom_alignment=0,1,2,3,5,4,6,7::1,2,4,3,0,5,6,7::1,2,3,0,4,5,6,7::2,1,3,4,5,6,0,7::1,2,3,4,7,6,0,5::7,0,4,3,2,5,6,1::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f5342d1fdf15744c668c0a01ae75cffc
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6531 --custom_alignment=0,1,2,3,5,4,6,7::1,2,4,3,0,5,6,7::1,2,3,0,4,5,6,7::2,1,3,4,5,6,0,7::1,2,3,4,7,6,0,5::7,0,4,3,2,5,6,1::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f5342d1fdf15744c668c0a01ae75cffc
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7004 --custom_alignment=0,1,2,3,5,4,6,7::1,2,4,3,0,5,6,7::1,2,3,0,4,5,6,7::2,1,3,4,5,6,0,7::1,2,3,4,7,6,0,5::7,0,4,3,2,5,6,1::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f5342d1fdf15744c668c0a01ae75cffc
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7415 --custom_alignment=0,1,2,3,5,4,6,7::1,2,4,3,0,5,6,7::1,2,3,0,4,5,6,7::2,1,3,4,5,6,0,7::1,2,3,4,7,6,0,5::7,0,4,3,2,5,6,1::2,0,1,3,4,5,6,7::2,6,0,1,4,5,3,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 7,1,2,3,4,0,6,5::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,5,4,3,7,6,0::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,0,1,3,4,5,6,7::2,0,3,1,4,5,6,7

# 0869768e18baf67df743e2538c1a00cd
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6243 --custom_alignment=7,1,2,3,4,0,6,5::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,5,4,3,7,6,0::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,0,1,3,4,5,6,7::2,0,3,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0869768e18baf67df743e2538c1a00cd
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=1981 --custom_alignment=7,1,2,3,4,0,6,5::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,5,4,3,7,6,0::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,0,1,3,4,5,6,7::2,0,3,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0869768e18baf67df743e2538c1a00cd
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4139 --custom_alignment=7,1,2,3,4,0,6,5::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,5,4,3,7,6,0::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,0,1,3,4,5,6,7::2,0,3,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0869768e18baf67df743e2538c1a00cd
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4170 --custom_alignment=7,1,2,3,4,0,6,5::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,5,4,3,7,6,0::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,0,1,3,4,5,6,7::2,0,3,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0869768e18baf67df743e2538c1a00cd
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=151 --custom_alignment=7,1,2,3,4,0,6,5::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,5,4,3,7,6,0::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,0,1,3,4,5,6,7::2,0,3,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 4,1,3,2,5,0,7,6::1,2,0,3,4,5,6,7::1,2,3,5,4,0,6,7::1,2,0,4,5,3,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 62fe719074fc13bd2543f7f2bb21173b
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4623 --custom_alignment=4,1,3,2,5,0,7,6::1,2,0,3,4,5,6,7::1,2,3,5,4,0,6,7::1,2,0,4,5,3,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 62fe719074fc13bd2543f7f2bb21173b
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5103 --custom_alignment=4,1,3,2,5,0,7,6::1,2,0,3,4,5,6,7::1,2,3,5,4,0,6,7::1,2,0,4,5,3,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 62fe719074fc13bd2543f7f2bb21173b
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=1275 --custom_alignment=4,1,3,2,5,0,7,6::1,2,0,3,4,5,6,7::1,2,3,5,4,0,6,7::1,2,0,4,5,3,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 62fe719074fc13bd2543f7f2bb21173b
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4037 --custom_alignment=4,1,3,2,5,0,7,6::1,2,0,3,4,5,6,7::1,2,3,5,4,0,6,7::1,2,0,4,5,3,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 62fe719074fc13bd2543f7f2bb21173b
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4884 --custom_alignment=4,1,3,2,5,0,7,6::1,2,0,3,4,5,6,7::1,2,3,5,4,0,6,7::1,2,0,4,5,3,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 4,1,2,3,0,5,6,7::1,7,0,3,4,5,6,2::1,3,2,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,4,3,1,5,6,7::2,3,4,1,0,7,6,5

# 9036c1d696408e615c81bf82277be86a
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=8882 --custom_alignment=4,1,2,3,0,5,6,7::1,7,0,3,4,5,6,2::1,3,2,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,4,3,1,5,6,7::2,3,4,1,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9036c1d696408e615c81bf82277be86a
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6958 --custom_alignment=4,1,2,3,0,5,6,7::1,7,0,3,4,5,6,2::1,3,2,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,4,3,1,5,6,7::2,3,4,1,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9036c1d696408e615c81bf82277be86a
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5169 --custom_alignment=4,1,2,3,0,5,6,7::1,7,0,3,4,5,6,2::1,3,2,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,4,3,1,5,6,7::2,3,4,1,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9036c1d696408e615c81bf82277be86a
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7771 --custom_alignment=4,1,2,3,0,5,6,7::1,7,0,3,4,5,6,2::1,3,2,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,4,3,1,5,6,7::2,3,4,1,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9036c1d696408e615c81bf82277be86a
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6267 --custom_alignment=4,1,2,3,0,5,6,7::1,7,0,3,4,5,6,2::1,3,2,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,4,3,1,5,6,7::2,3,4,1,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,1,2,3,4,7,6,5::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,2,3,4,5,6,0,7::6,0,2,3,4,5,1,7::2,4,1,3,6,5,7,0::1,3,0,2,4,5,6,7

# cab30a899be1157cf1fe748cc597d2bc
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5465 --custom_alignment=0,1,2,3,4,7,6,5::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,2,3,4,5,6,0,7::6,0,2,3,4,5,1,7::2,4,1,3,6,5,7,0::1,3,0,2,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# cab30a899be1157cf1fe748cc597d2bc
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=260 --custom_alignment=0,1,2,3,4,7,6,5::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,2,3,4,5,6,0,7::6,0,2,3,4,5,1,7::2,4,1,3,6,5,7,0::1,3,0,2,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# cab30a899be1157cf1fe748cc597d2bc
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4922 --custom_alignment=0,1,2,3,4,7,6,5::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,2,3,4,5,6,0,7::6,0,2,3,4,5,1,7::2,4,1,3,6,5,7,0::1,3,0,2,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# cab30a899be1157cf1fe748cc597d2bc
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2683 --custom_alignment=0,1,2,3,4,7,6,5::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,2,3,4,5,6,0,7::6,0,2,3,4,5,1,7::2,4,1,3,6,5,7,0::1,3,0,2,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# cab30a899be1157cf1fe748cc597d2bc
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2726 --custom_alignment=0,1,2,3,4,7,6,5::3,2,0,1,4,5,6,7::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,2,3,4,5,6,0,7::6,0,2,3,4,5,1,7::2,4,1,3,6,5,7,0::1,3,0,2,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::3,7,6,0,4,5,1,2::1,2,3,0,5,4,6,7::1,3,2,4,5,6,0,7::1,0,2,3,6,5,4,7::2,0,1,3,4,5,6,7::2,4,0,1,3,5,6,7

# 4bc3bb22a6d50a8088c37debceba0018
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4621 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::3,7,6,0,4,5,1,2::1,2,3,0,5,4,6,7::1,3,2,4,5,6,0,7::1,0,2,3,6,5,4,7::2,0,1,3,4,5,6,7::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4bc3bb22a6d50a8088c37debceba0018
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7632 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::3,7,6,0,4,5,1,2::1,2,3,0,5,4,6,7::1,3,2,4,5,6,0,7::1,0,2,3,6,5,4,7::2,0,1,3,4,5,6,7::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4bc3bb22a6d50a8088c37debceba0018
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2051 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::3,7,6,0,4,5,1,2::1,2,3,0,5,4,6,7::1,3,2,4,5,6,0,7::1,0,2,3,6,5,4,7::2,0,1,3,4,5,6,7::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4bc3bb22a6d50a8088c37debceba0018
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=47 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::3,7,6,0,4,5,1,2::1,2,3,0,5,4,6,7::1,3,2,4,5,6,0,7::1,0,2,3,6,5,4,7::2,0,1,3,4,5,6,7::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 4bc3bb22a6d50a8088c37debceba0018
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=9369 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::3,7,6,0,4,5,1,2::1,2,3,0,5,4,6,7::1,3,2,4,5,6,0,7::1,0,2,3,6,5,4,7::2,0,1,3,4,5,6,7::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,0,7,6::1,2,3,4,5,6,7,0::1,0,2,3,4,5,6,7::2,0,6,3,4,5,1,7::3,4,0,1,2,5,6,7

# 3c7cd57ac3e5bf376161d442ec843a42
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=8329 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,0,7,6::1,2,3,4,5,6,7,0::1,0,2,3,4,5,6,7::2,0,6,3,4,5,1,7::3,4,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3c7cd57ac3e5bf376161d442ec843a42
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=3054 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,0,7,6::1,2,3,4,5,6,7,0::1,0,2,3,4,5,6,7::2,0,6,3,4,5,1,7::3,4,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3c7cd57ac3e5bf376161d442ec843a42
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5272 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,0,7,6::1,2,3,4,5,6,7,0::1,0,2,3,4,5,6,7::2,0,6,3,4,5,1,7::3,4,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3c7cd57ac3e5bf376161d442ec843a42
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=1155 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,0,7,6::1,2,3,4,5,6,7,0::1,0,2,3,4,5,6,7::2,0,6,3,4,5,1,7::3,4,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3c7cd57ac3e5bf376161d442ec843a42
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4216 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,0,7,6::1,2,3,4,5,6,7,0::1,0,2,3,4,5,6,7::2,0,6,3,4,5,1,7::3,4,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,5,2,3,4,1,6,7::1,3,0,2,5,4,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,1,6,0::6,2,3,4,5,1,0,7::1,0,3,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# f0565923990fe489629bcb5232adb3d5
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=9072 --custom_alignment=0,5,2,3,4,1,6,7::1,3,0,2,5,4,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,1,6,0::6,2,3,4,5,1,0,7::1,0,3,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f0565923990fe489629bcb5232adb3d5
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5411 --custom_alignment=0,5,2,3,4,1,6,7::1,3,0,2,5,4,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,1,6,0::6,2,3,4,5,1,0,7::1,0,3,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f0565923990fe489629bcb5232adb3d5
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=3364 --custom_alignment=0,5,2,3,4,1,6,7::1,3,0,2,5,4,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,1,6,0::6,2,3,4,5,1,0,7::1,0,3,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f0565923990fe489629bcb5232adb3d5
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7511 --custom_alignment=0,5,2,3,4,1,6,7::1,3,0,2,5,4,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,1,6,0::6,2,3,4,5,1,0,7::1,0,3,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f0565923990fe489629bcb5232adb3d5
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4734 --custom_alignment=0,5,2,3,4,1,6,7::1,3,0,2,5,4,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,1,6,0::6,2,3,4,5,1,0,7::1,0,3,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,3,2,1,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,0,5,6,7::1,2,3,4,0,6,5,7::1,2,0,3,4,5,6,7::4,5,1,2,0,3,6,7::2,3,0,1,4,5,6,7

# 122a76a2b52f29765796d7aae084c892
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=1033 --custom_alignment=0,3,2,1,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,0,5,6,7::1,2,3,4,0,6,5,7::1,2,0,3,4,5,6,7::4,5,1,2,0,3,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 122a76a2b52f29765796d7aae084c892
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5754 --custom_alignment=0,3,2,1,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,0,5,6,7::1,2,3,4,0,6,5,7::1,2,0,3,4,5,6,7::4,5,1,2,0,3,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 122a76a2b52f29765796d7aae084c892
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=3274 --custom_alignment=0,3,2,1,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,0,5,6,7::1,2,3,4,0,6,5,7::1,2,0,3,4,5,6,7::4,5,1,2,0,3,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 122a76a2b52f29765796d7aae084c892
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4702 --custom_alignment=0,3,2,1,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,0,5,6,7::1,2,3,4,0,6,5,7::1,2,0,3,4,5,6,7::4,5,1,2,0,3,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 122a76a2b52f29765796d7aae084c892
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=1588 --custom_alignment=0,3,2,1,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,0,5,6,7::1,2,3,4,0,6,5,7::1,2,0,3,4,5,6,7::4,5,1,2,0,3,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 2,1,0,3,4,5,6,7::6,5,0,3,4,2,1,7::1,2,3,0,4,5,6,7::1,2,3,5,4,0,6,7::5,3,2,4,1,6,0,7::5,0,2,3,4,1,6,7::2,0,1,3,4,5,6,7::3,2,0,1,4,5,6,7

# 56acd7d427a9f0756e9d6392b1e68d69
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2667 --custom_alignment=2,1,0,3,4,5,6,7::6,5,0,3,4,2,1,7::1,2,3,0,4,5,6,7::1,2,3,5,4,0,6,7::5,3,2,4,1,6,0,7::5,0,2,3,4,1,6,7::2,0,1,3,4,5,6,7::3,2,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 56acd7d427a9f0756e9d6392b1e68d69
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6713 --custom_alignment=2,1,0,3,4,5,6,7::6,5,0,3,4,2,1,7::1,2,3,0,4,5,6,7::1,2,3,5,4,0,6,7::5,3,2,4,1,6,0,7::5,0,2,3,4,1,6,7::2,0,1,3,4,5,6,7::3,2,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 56acd7d427a9f0756e9d6392b1e68d69
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6571 --custom_alignment=2,1,0,3,4,5,6,7::6,5,0,3,4,2,1,7::1,2,3,0,4,5,6,7::1,2,3,5,4,0,6,7::5,3,2,4,1,6,0,7::5,0,2,3,4,1,6,7::2,0,1,3,4,5,6,7::3,2,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 56acd7d427a9f0756e9d6392b1e68d69
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6648 --custom_alignment=2,1,0,3,4,5,6,7::6,5,0,3,4,2,1,7::1,2,3,0,4,5,6,7::1,2,3,5,4,0,6,7::5,3,2,4,1,6,0,7::5,0,2,3,4,1,6,7::2,0,1,3,4,5,6,7::3,2,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 56acd7d427a9f0756e9d6392b1e68d69
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=3484 --custom_alignment=2,1,0,3,4,5,6,7::6,5,0,3,4,2,1,7::1,2,3,0,4,5,6,7::1,2,3,5,4,0,6,7::5,3,2,4,1,6,0,7::5,0,2,3,4,1,6,7::2,0,1,3,4,5,6,7::3,2,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,1,2,3,5,4,6,7::5,7,0,3,4,1,6,2::1,2,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,5,3,4,2,6,0,7::1,0,4,3,2,5,6,7::2,0,1,3,4,5,6,7::0,3,2,1,5,4,6,7

# 5e500969cee22c3e9d2bc3440c052a06
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=9913 --custom_alignment=0,1,2,3,5,4,6,7::5,7,0,3,4,1,6,2::1,2,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,5,3,4,2,6,0,7::1,0,4,3,2,5,6,7::2,0,1,3,4,5,6,7::0,3,2,1,5,4,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 5e500969cee22c3e9d2bc3440c052a06
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=8080 --custom_alignment=0,1,2,3,5,4,6,7::5,7,0,3,4,1,6,2::1,2,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,5,3,4,2,6,0,7::1,0,4,3,2,5,6,7::2,0,1,3,4,5,6,7::0,3,2,1,5,4,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 5e500969cee22c3e9d2bc3440c052a06
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5924 --custom_alignment=0,1,2,3,5,4,6,7::5,7,0,3,4,1,6,2::1,2,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,5,3,4,2,6,0,7::1,0,4,3,2,5,6,7::2,0,1,3,4,5,6,7::0,3,2,1,5,4,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 5e500969cee22c3e9d2bc3440c052a06
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=1405 --custom_alignment=0,1,2,3,5,4,6,7::5,7,0,3,4,1,6,2::1,2,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,5,3,4,2,6,0,7::1,0,4,3,2,5,6,7::2,0,1,3,4,5,6,7::0,3,2,1,5,4,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 5e500969cee22c3e9d2bc3440c052a06
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=8001 --custom_alignment=0,1,2,3,5,4,6,7::5,7,0,3,4,1,6,2::1,2,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,5,3,4,2,6,0,7::1,0,4,3,2,5,6,7::2,0,1,3,4,5,6,7::0,3,2,1,5,4,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,1,2,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,6,0,4,5,3,7::1,6,3,4,0,5,2,7::1,2,3,4,5,6,0,7::1,5,2,3,4,0,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7

# 82b50cb9e0d076b9d69d0894c6710564
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7423 --custom_alignment=0,1,2,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,6,0,4,5,3,7::1,6,3,4,0,5,2,7::1,2,3,4,5,6,0,7::1,5,2,3,4,0,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 82b50cb9e0d076b9d69d0894c6710564
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2829 --custom_alignment=0,1,2,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,6,0,4,5,3,7::1,6,3,4,0,5,2,7::1,2,3,4,5,6,0,7::1,5,2,3,4,0,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 82b50cb9e0d076b9d69d0894c6710564
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7697 --custom_alignment=0,1,2,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,6,0,4,5,3,7::1,6,3,4,0,5,2,7::1,2,3,4,5,6,0,7::1,5,2,3,4,0,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 82b50cb9e0d076b9d69d0894c6710564
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=1291 --custom_alignment=0,1,2,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,6,0,4,5,3,7::1,6,3,4,0,5,2,7::1,2,3,4,5,6,0,7::1,5,2,3,4,0,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 82b50cb9e0d076b9d69d0894c6710564
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=3806 --custom_alignment=0,1,2,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,6,0,4,5,3,7::1,6,3,4,0,5,2,7::1,2,3,4,5,6,0,7::1,5,2,3,4,0,6,7::2,0,1,3,5,4,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,1,6,2,4,5,3,7::1,2,4,3,0,5,6,7::1,2,5,0,6,4,3,7::1,2,3,4,5,0,6,7::1,4,3,2,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7

# 2ce245f522305a3f899c37ec74ddb0b1
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=295 --custom_alignment=0,1,6,2,4,5,3,7::1,2,4,3,0,5,6,7::1,2,5,0,6,4,3,7::1,2,3,4,5,0,6,7::1,4,3,2,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 2ce245f522305a3f899c37ec74ddb0b1
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5387 --custom_alignment=0,1,6,2,4,5,3,7::1,2,4,3,0,5,6,7::1,2,5,0,6,4,3,7::1,2,3,4,5,0,6,7::1,4,3,2,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 2ce245f522305a3f899c37ec74ddb0b1
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4448 --custom_alignment=0,1,6,2,4,5,3,7::1,2,4,3,0,5,6,7::1,2,5,0,6,4,3,7::1,2,3,4,5,0,6,7::1,4,3,2,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 2ce245f522305a3f899c37ec74ddb0b1
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7587 --custom_alignment=0,1,6,2,4,5,3,7::1,2,4,3,0,5,6,7::1,2,5,0,6,4,3,7::1,2,3,4,5,0,6,7::1,4,3,2,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 2ce245f522305a3f899c37ec74ddb0b1
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=8675 --custom_alignment=0,1,6,2,4,5,3,7::1,2,4,3,0,5,6,7::1,2,5,0,6,4,3,7::1,2,3,4,5,0,6,7::1,4,3,2,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,6,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 5,1,2,3,4,0,6,7::3,2,0,1,4,5,6,7::1,2,5,0,4,3,6,7::1,2,5,4,3,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,1,0,3,7,5,6,4::2,3,0,1,4,5,7,6

# 0ff1efbaf66c29eed4f656c291fe2621
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4527 --custom_alignment=5,1,2,3,4,0,6,7::3,2,0,1,4,5,6,7::1,2,5,0,4,3,6,7::1,2,5,4,3,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,1,0,3,7,5,6,4::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0ff1efbaf66c29eed4f656c291fe2621
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=9 --custom_alignment=5,1,2,3,4,0,6,7::3,2,0,1,4,5,6,7::1,2,5,0,4,3,6,7::1,2,5,4,3,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,1,0,3,7,5,6,4::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0ff1efbaf66c29eed4f656c291fe2621
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2768 --custom_alignment=5,1,2,3,4,0,6,7::3,2,0,1,4,5,6,7::1,2,5,0,4,3,6,7::1,2,5,4,3,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,1,0,3,7,5,6,4::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0ff1efbaf66c29eed4f656c291fe2621
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7672 --custom_alignment=5,1,2,3,4,0,6,7::3,2,0,1,4,5,6,7::1,2,5,0,4,3,6,7::1,2,5,4,3,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,1,0,3,7,5,6,4::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0ff1efbaf66c29eed4f656c291fe2621
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=1636 --custom_alignment=5,1,2,3,4,0,6,7::3,2,0,1,4,5,6,7::1,2,5,0,4,3,6,7::1,2,5,4,3,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,1,0,3,7,5,6,4::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 6,1,2,7,4,5,0,3::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,5,3,4,2,0,6,7::1,2,6,4,5,7,0,3::1,6,2,3,4,0,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 785ec28e0ef3a519e7f6cc9e469a6ca8
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4674 --custom_alignment=6,1,2,7,4,5,0,3::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,5,3,4,2,0,6,7::1,2,6,4,5,7,0,3::1,6,2,3,4,0,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 785ec28e0ef3a519e7f6cc9e469a6ca8
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6614 --custom_alignment=6,1,2,7,4,5,0,3::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,5,3,4,2,0,6,7::1,2,6,4,5,7,0,3::1,6,2,3,4,0,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 785ec28e0ef3a519e7f6cc9e469a6ca8
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2932 --custom_alignment=6,1,2,7,4,5,0,3::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,5,3,4,2,0,6,7::1,2,6,4,5,7,0,3::1,6,2,3,4,0,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 785ec28e0ef3a519e7f6cc9e469a6ca8
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6535 --custom_alignment=6,1,2,7,4,5,0,3::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,5,3,4,2,0,6,7::1,2,6,4,5,7,0,3::1,6,2,3,4,0,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 785ec28e0ef3a519e7f6cc9e469a6ca8
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4102 --custom_alignment=6,1,2,7,4,5,0,3::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,5,3,4,2,0,6,7::1,2,6,4,5,7,0,3::1,6,2,3,4,0,5,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,1,2,3,4,5,6,7::6,2,1,3,7,5,0,4::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,7,3,4,5,6,0,2::2,0,1,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 3d7f5b268105a77a0b53d7a62613a1f1
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=8397 --custom_alignment=0,1,2,3,4,5,6,7::6,2,1,3,7,5,0,4::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,7,3,4,5,6,0,2::2,0,1,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3d7f5b268105a77a0b53d7a62613a1f1
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=9897 --custom_alignment=0,1,2,3,4,5,6,7::6,2,1,3,7,5,0,4::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,7,3,4,5,6,0,2::2,0,1,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3d7f5b268105a77a0b53d7a62613a1f1
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=8494 --custom_alignment=0,1,2,3,4,5,6,7::6,2,1,3,7,5,0,4::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,7,3,4,5,6,0,2::2,0,1,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3d7f5b268105a77a0b53d7a62613a1f1
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=9968 --custom_alignment=0,1,2,3,4,5,6,7::6,2,1,3,7,5,0,4::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,7,3,4,5,6,0,2::2,0,1,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3d7f5b268105a77a0b53d7a62613a1f1
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=276 --custom_alignment=0,1,2,3,4,5,6,7::6,2,1,3,7,5,0,4::1,2,3,0,4,5,6,7::5,2,3,4,1,0,6,7::1,7,3,4,5,6,0,2::2,0,1,3,4,5,6,7::0,2,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,1,2,3,4,5,6,7::2,1,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,6,1,3,4,5,7,0::2,3,0,1,4,5,7,6

# c045fc79f992aabf097abc7dd31762b3
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=8706 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,6,1,3,4,5,7,0::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c045fc79f992aabf097abc7dd31762b3
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=533 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,6,1,3,4,5,7,0::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c045fc79f992aabf097abc7dd31762b3
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5810 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,6,1,3,4,5,7,0::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c045fc79f992aabf097abc7dd31762b3
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=7101 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,6,1,3,4,5,7,0::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c045fc79f992aabf097abc7dd31762b3
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6521 --custom_alignment=0,1,2,3,4,5,6,7::2,1,0,3,7,5,6,4::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,7,4,5,6,3::2,6,1,3,4,5,7,0::2,3,0,1,4,5,7,6 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,5,2,3,4,1,6,7::1,2,4,3,0,5,6,7::3,2,1,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,7,6,5::2,0,6,5,4,3,1,7::2,3,0,1,4,5,6,7

# 24cf7f9cfcbc2ae2dd2945a5cb647946
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=4615 --custom_alignment=0,5,2,3,4,1,6,7::1,2,4,3,0,5,6,7::3,2,1,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,7,6,5::2,0,6,5,4,3,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24cf7f9cfcbc2ae2dd2945a5cb647946
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5640 --custom_alignment=0,5,2,3,4,1,6,7::1,2,4,3,0,5,6,7::3,2,1,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,7,6,5::2,0,6,5,4,3,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24cf7f9cfcbc2ae2dd2945a5cb647946
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2667 --custom_alignment=0,5,2,3,4,1,6,7::1,2,4,3,0,5,6,7::3,2,1,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,7,6,5::2,0,6,5,4,3,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24cf7f9cfcbc2ae2dd2945a5cb647946
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=8310 --custom_alignment=0,5,2,3,4,1,6,7::1,2,4,3,0,5,6,7::3,2,1,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,7,6,5::2,0,6,5,4,3,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24cf7f9cfcbc2ae2dd2945a5cb647946
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=8231 --custom_alignment=0,5,2,3,4,1,6,7::1,2,4,3,0,5,6,7::3,2,1,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,7,6,5::2,0,6,5,4,3,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 8 alignment for 9xx: 0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::7,3,2,0,4,5,6,1::1,2,3,4,7,0,6,5::1,2,3,4,5,6,0,7::1,0,5,3,4,7,6,2::6,0,1,3,4,5,2,7::2,3,0,1,4,5,6,7

# 3d282e06ab88423aea75b10a58432701
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2535 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::7,3,2,0,4,5,6,1::1,2,3,4,7,0,6,5::1,2,3,4,5,6,0,7::1,0,5,3,4,7,6,2::6,0,1,3,4,5,2,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3d282e06ab88423aea75b10a58432701
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=5333 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::7,3,2,0,4,5,6,1::1,2,3,4,7,0,6,5::1,2,3,4,5,6,0,7::1,0,5,3,4,7,6,2::6,0,1,3,4,5,2,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3d282e06ab88423aea75b10a58432701
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=2484 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::7,3,2,0,4,5,6,1::1,2,3,4,7,0,6,5::1,2,3,4,5,6,0,7::1,0,5,3,4,7,6,2::6,0,1,3,4,5,2,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3d282e06ab88423aea75b10a58432701
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=9249 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::7,3,2,0,4,5,6,1::1,2,3,4,7,0,6,5::1,2,3,4,5,6,0,7::1,0,5,3,4,7,6,2::6,0,1,3,4,5,2,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3d282e06ab88423aea75b10a58432701
sbatch -J search_9xx_mutate_8 submit.sh python 1.train.py --seed=6401 --custom_alignment=0,1,6,3,4,5,2,7::1,2,0,3,4,5,6,7::7,3,2,0,4,5,6,1::1,2,3,4,7,0,6,5::1,2,3,4,5,6,0,7::1,0,5,3,4,7,6,2::6,0,1,3,4,5,2,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_8 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,1,2,3,4,5,7,6::3,2,0,1,4,5,6,7::7,2,3,0,4,5,6,1::1,0,3,2,5,4,6,7::5,2,4,3,1,6,0,7::1,0,2,3,4,5,6,7::2,0,1,5,4,3,6,7::4,2,3,1,0,5,6,7

# 29b6d663eb05fa83a29e061c2083a940
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=6767 --custom_alignment=0,1,2,3,4,5,7,6::3,2,0,1,4,5,6,7::7,2,3,0,4,5,6,1::1,0,3,2,5,4,6,7::5,2,4,3,1,6,0,7::1,0,2,3,4,5,6,7::2,0,1,5,4,3,6,7::4,2,3,1,0,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 29b6d663eb05fa83a29e061c2083a940
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4597 --custom_alignment=0,1,2,3,4,5,7,6::3,2,0,1,4,5,6,7::7,2,3,0,4,5,6,1::1,0,3,2,5,4,6,7::5,2,4,3,1,6,0,7::1,0,2,3,4,5,6,7::2,0,1,5,4,3,6,7::4,2,3,1,0,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 29b6d663eb05fa83a29e061c2083a940
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=6064 --custom_alignment=0,1,2,3,4,5,7,6::3,2,0,1,4,5,6,7::7,2,3,0,4,5,6,1::1,0,3,2,5,4,6,7::5,2,4,3,1,6,0,7::1,0,2,3,4,5,6,7::2,0,1,5,4,3,6,7::4,2,3,1,0,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 29b6d663eb05fa83a29e061c2083a940
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1447 --custom_alignment=0,1,2,3,4,5,7,6::3,2,0,1,4,5,6,7::7,2,3,0,4,5,6,1::1,0,3,2,5,4,6,7::5,2,4,3,1,6,0,7::1,0,2,3,4,5,6,7::2,0,1,5,4,3,6,7::4,2,3,1,0,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 29b6d663eb05fa83a29e061c2083a940
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9530 --custom_alignment=0,1,2,3,4,5,7,6::3,2,0,1,4,5,6,7::7,2,3,0,4,5,6,1::1,0,3,2,5,4,6,7::5,2,4,3,1,6,0,7::1,0,2,3,4,5,6,7::2,0,1,5,4,3,6,7::4,2,3,1,0,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,1,4,3,2,5,7,6::6,2,4,3,0,5,1,7::1,2,3,0,4,5,6,7::1,2,5,4,3,6,0,7::1,6,3,4,5,2,0,7::1,6,4,3,2,7,0,5::2,0,1,3,4,5,6,7::2,7,0,4,1,5,6,3

# 129d9ffa5666bb023c570bc2cf300624
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4044 --custom_alignment=0,1,4,3,2,5,7,6::6,2,4,3,0,5,1,7::1,2,3,0,4,5,6,7::1,2,5,4,3,6,0,7::1,6,3,4,5,2,0,7::1,6,4,3,2,7,0,5::2,0,1,3,4,5,6,7::2,7,0,4,1,5,6,3 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 129d9ffa5666bb023c570bc2cf300624
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4873 --custom_alignment=0,1,4,3,2,5,7,6::6,2,4,3,0,5,1,7::1,2,3,0,4,5,6,7::1,2,5,4,3,6,0,7::1,6,3,4,5,2,0,7::1,6,4,3,2,7,0,5::2,0,1,3,4,5,6,7::2,7,0,4,1,5,6,3 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 129d9ffa5666bb023c570bc2cf300624
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8296 --custom_alignment=0,1,4,3,2,5,7,6::6,2,4,3,0,5,1,7::1,2,3,0,4,5,6,7::1,2,5,4,3,6,0,7::1,6,3,4,5,2,0,7::1,6,4,3,2,7,0,5::2,0,1,3,4,5,6,7::2,7,0,4,1,5,6,3 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 129d9ffa5666bb023c570bc2cf300624
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3959 --custom_alignment=0,1,4,3,2,5,7,6::6,2,4,3,0,5,1,7::1,2,3,0,4,5,6,7::1,2,5,4,3,6,0,7::1,6,3,4,5,2,0,7::1,6,4,3,2,7,0,5::2,0,1,3,4,5,6,7::2,7,0,4,1,5,6,3 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 129d9ffa5666bb023c570bc2cf300624
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1842 --custom_alignment=0,1,4,3,2,5,7,6::6,2,4,3,0,5,1,7::1,2,3,0,4,5,6,7::1,2,5,4,3,6,0,7::1,6,3,4,5,2,0,7::1,6,4,3,2,7,0,5::2,0,1,3,4,5,6,7::2,7,0,4,1,5,6,3 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,2,1,3,4,7,6,5::5,2,0,3,4,1,6,7::1,2,3,7,4,5,6,0::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,6,4,5,3,7::2,3,7,1,6,5,4,0

# ac159a5360f48f4729eacf990622df82
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3036 --custom_alignment=0,2,1,3,4,7,6,5::5,2,0,3,4,1,6,7::1,2,3,7,4,5,6,0::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,6,4,5,3,7::2,3,7,1,6,5,4,0 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ac159a5360f48f4729eacf990622df82
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9873 --custom_alignment=0,2,1,3,4,7,6,5::5,2,0,3,4,1,6,7::1,2,3,7,4,5,6,0::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,6,4,5,3,7::2,3,7,1,6,5,4,0 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ac159a5360f48f4729eacf990622df82
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=6817 --custom_alignment=0,2,1,3,4,7,6,5::5,2,0,3,4,1,6,7::1,2,3,7,4,5,6,0::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,6,4,5,3,7::2,3,7,1,6,5,4,0 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ac159a5360f48f4729eacf990622df82
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8046 --custom_alignment=0,2,1,3,4,7,6,5::5,2,0,3,4,1,6,7::1,2,3,7,4,5,6,0::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,6,4,5,3,7::2,3,7,1,6,5,4,0 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# ac159a5360f48f4729eacf990622df82
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=7980 --custom_alignment=0,2,1,3,4,7,6,5::5,2,0,3,4,1,6,7::1,2,3,7,4,5,6,0::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,3,2,0,4,5,6,7::2,0,1,6,4,5,3,7::2,3,7,1,6,5,4,0 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 2,4,6,3,0,5,1,7::3,2,4,1,5,0,6,7::5,2,3,0,4,1,6,7::1,7,4,3,5,0,2,6::1,2,3,4,5,6,0,7::3,4,7,1,0,5,6,2::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# 407fb3b15bf2e33b863834412e2a26fe
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=6939 --custom_alignment=2,4,6,3,0,5,1,7::3,2,4,1,5,0,6,7::5,2,3,0,4,1,6,7::1,7,4,3,5,0,2,6::1,2,3,4,5,6,0,7::3,4,7,1,0,5,6,2::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 407fb3b15bf2e33b863834412e2a26fe
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=7848 --custom_alignment=2,4,6,3,0,5,1,7::3,2,4,1,5,0,6,7::5,2,3,0,4,1,6,7::1,7,4,3,5,0,2,6::1,2,3,4,5,6,0,7::3,4,7,1,0,5,6,2::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 407fb3b15bf2e33b863834412e2a26fe
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3634 --custom_alignment=2,4,6,3,0,5,1,7::3,2,4,1,5,0,6,7::5,2,3,0,4,1,6,7::1,7,4,3,5,0,2,6::1,2,3,4,5,6,0,7::3,4,7,1,0,5,6,2::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 407fb3b15bf2e33b863834412e2a26fe
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8834 --custom_alignment=2,4,6,3,0,5,1,7::3,2,4,1,5,0,6,7::5,2,3,0,4,1,6,7::1,7,4,3,5,0,2,6::1,2,3,4,5,6,0,7::3,4,7,1,0,5,6,2::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 407fb3b15bf2e33b863834412e2a26fe
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8092 --custom_alignment=2,4,6,3,0,5,1,7::3,2,4,1,5,0,6,7::5,2,3,0,4,1,6,7::1,7,4,3,5,0,2,6::1,2,3,4,5,6,0,7::3,4,7,1,0,5,6,2::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 3,5,2,0,4,1,6,7::6,2,4,3,0,5,1,7::1,7,5,0,4,6,3,2::1,2,3,4,5,0,6,7::1,2,3,4,5,0,6,7::1,0,2,3,4,5,6,7::7,0,1,3,2,5,6,4::1,2,4,3,0,7,6,5

# eff9a1c092ca316a423ecb1cb6741764
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=5219 --custom_alignment=3,5,2,0,4,1,6,7::6,2,4,3,0,5,1,7::1,7,5,0,4,6,3,2::1,2,3,4,5,0,6,7::1,2,3,4,5,0,6,7::1,0,2,3,4,5,6,7::7,0,1,3,2,5,6,4::1,2,4,3,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# eff9a1c092ca316a423ecb1cb6741764
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8277 --custom_alignment=3,5,2,0,4,1,6,7::6,2,4,3,0,5,1,7::1,7,5,0,4,6,3,2::1,2,3,4,5,0,6,7::1,2,3,4,5,0,6,7::1,0,2,3,4,5,6,7::7,0,1,3,2,5,6,4::1,2,4,3,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# eff9a1c092ca316a423ecb1cb6741764
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3883 --custom_alignment=3,5,2,0,4,1,6,7::6,2,4,3,0,5,1,7::1,7,5,0,4,6,3,2::1,2,3,4,5,0,6,7::1,2,3,4,5,0,6,7::1,0,2,3,4,5,6,7::7,0,1,3,2,5,6,4::1,2,4,3,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# eff9a1c092ca316a423ecb1cb6741764
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8546 --custom_alignment=3,5,2,0,4,1,6,7::6,2,4,3,0,5,1,7::1,7,5,0,4,6,3,2::1,2,3,4,5,0,6,7::1,2,3,4,5,0,6,7::1,0,2,3,4,5,6,7::7,0,1,3,2,5,6,4::1,2,4,3,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# eff9a1c092ca316a423ecb1cb6741764
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=6576 --custom_alignment=3,5,2,0,4,1,6,7::6,2,4,3,0,5,1,7::1,7,5,0,4,6,3,2::1,2,3,4,5,0,6,7::1,2,3,4,5,0,6,7::1,0,2,3,4,5,6,7::7,0,1,3,2,5,6,4::1,2,4,3,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,1,2,3,4,6,7,5::1,2,0,3,6,5,4,7::1,2,3,0,4,5,6,7::1,6,3,4,7,0,2,5::1,0,3,4,5,6,2,7::3,0,2,1,4,5,6,7::6,3,1,7,5,4,2,0::2,0,3,1,4,7,6,5

# 24afd3f917b2cf0022cf3c4ec5b2a15c
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=6052 --custom_alignment=0,1,2,3,4,6,7,5::1,2,0,3,6,5,4,7::1,2,3,0,4,5,6,7::1,6,3,4,7,0,2,5::1,0,3,4,5,6,2,7::3,0,2,1,4,5,6,7::6,3,1,7,5,4,2,0::2,0,3,1,4,7,6,5 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24afd3f917b2cf0022cf3c4ec5b2a15c
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4957 --custom_alignment=0,1,2,3,4,6,7,5::1,2,0,3,6,5,4,7::1,2,3,0,4,5,6,7::1,6,3,4,7,0,2,5::1,0,3,4,5,6,2,7::3,0,2,1,4,5,6,7::6,3,1,7,5,4,2,0::2,0,3,1,4,7,6,5 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24afd3f917b2cf0022cf3c4ec5b2a15c
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8584 --custom_alignment=0,1,2,3,4,6,7,5::1,2,0,3,6,5,4,7::1,2,3,0,4,5,6,7::1,6,3,4,7,0,2,5::1,0,3,4,5,6,2,7::3,0,2,1,4,5,6,7::6,3,1,7,5,4,2,0::2,0,3,1,4,7,6,5 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24afd3f917b2cf0022cf3c4ec5b2a15c
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8092 --custom_alignment=0,1,2,3,4,6,7,5::1,2,0,3,6,5,4,7::1,2,3,0,4,5,6,7::1,6,3,4,7,0,2,5::1,0,3,4,5,6,2,7::3,0,2,1,4,5,6,7::6,3,1,7,5,4,2,0::2,0,3,1,4,7,6,5 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24afd3f917b2cf0022cf3c4ec5b2a15c
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8228 --custom_alignment=0,1,2,3,4,6,7,5::1,2,0,3,6,5,4,7::1,2,3,0,4,5,6,7::1,6,3,4,7,0,2,5::1,0,3,4,5,6,2,7::3,0,2,1,4,5,6,7::6,3,1,7,5,4,2,0::2,0,3,1,4,7,6,5 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,7,2,3,4,5,6,1::1,3,7,5,6,2,4,0::1,7,2,0,4,3,6,5::1,5,3,4,0,2,6,7::3,2,1,4,5,6,0,7::6,0,2,7,4,5,1,3::2,0,6,3,4,5,1,7::2,3,5,1,4,0,6,7

# 3714e74adb2608702971deaf7e2a62e5
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4369 --custom_alignment=0,7,2,3,4,5,6,1::1,3,7,5,6,2,4,0::1,7,2,0,4,3,6,5::1,5,3,4,0,2,6,7::3,2,1,4,5,6,0,7::6,0,2,7,4,5,1,3::2,0,6,3,4,5,1,7::2,3,5,1,4,0,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3714e74adb2608702971deaf7e2a62e5
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3067 --custom_alignment=0,7,2,3,4,5,6,1::1,3,7,5,6,2,4,0::1,7,2,0,4,3,6,5::1,5,3,4,0,2,6,7::3,2,1,4,5,6,0,7::6,0,2,7,4,5,1,3::2,0,6,3,4,5,1,7::2,3,5,1,4,0,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3714e74adb2608702971deaf7e2a62e5
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=5560 --custom_alignment=0,7,2,3,4,5,6,1::1,3,7,5,6,2,4,0::1,7,2,0,4,3,6,5::1,5,3,4,0,2,6,7::3,2,1,4,5,6,0,7::6,0,2,7,4,5,1,3::2,0,6,3,4,5,1,7::2,3,5,1,4,0,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3714e74adb2608702971deaf7e2a62e5
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9203 --custom_alignment=0,7,2,3,4,5,6,1::1,3,7,5,6,2,4,0::1,7,2,0,4,3,6,5::1,5,3,4,0,2,6,7::3,2,1,4,5,6,0,7::6,0,2,7,4,5,1,3::2,0,6,3,4,5,1,7::2,3,5,1,4,0,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 3714e74adb2608702971deaf7e2a62e5
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1869 --custom_alignment=0,7,2,3,4,5,6,1::1,3,7,5,6,2,4,0::1,7,2,0,4,3,6,5::1,5,3,4,0,2,6,7::3,2,1,4,5,6,0,7::6,0,2,7,4,5,1,3::2,0,6,3,4,5,1,7::2,3,5,1,4,0,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,1,2,5,4,3,6,7::0,2,5,3,1,4,6,7::1,2,3,0,4,5,6,7::5,2,3,7,1,0,6,4::1,2,3,4,0,6,7,5::1,0,2,3,4,5,6,7::0,7,1,3,4,5,6,2::5,3,6,4,1,0,2,7

# df5a92a77db899799b71ec75d1970343
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1842 --custom_alignment=0,1,2,5,4,3,6,7::0,2,5,3,1,4,6,7::1,2,3,0,4,5,6,7::5,2,3,7,1,0,6,4::1,2,3,4,0,6,7,5::1,0,2,3,4,5,6,7::0,7,1,3,4,5,6,2::5,3,6,4,1,0,2,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# df5a92a77db899799b71ec75d1970343
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1839 --custom_alignment=0,1,2,5,4,3,6,7::0,2,5,3,1,4,6,7::1,2,3,0,4,5,6,7::5,2,3,7,1,0,6,4::1,2,3,4,0,6,7,5::1,0,2,3,4,5,6,7::0,7,1,3,4,5,6,2::5,3,6,4,1,0,2,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# df5a92a77db899799b71ec75d1970343
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9185 --custom_alignment=0,1,2,5,4,3,6,7::0,2,5,3,1,4,6,7::1,2,3,0,4,5,6,7::5,2,3,7,1,0,6,4::1,2,3,4,0,6,7,5::1,0,2,3,4,5,6,7::0,7,1,3,4,5,6,2::5,3,6,4,1,0,2,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# df5a92a77db899799b71ec75d1970343
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=330 --custom_alignment=0,1,2,5,4,3,6,7::0,2,5,3,1,4,6,7::1,2,3,0,4,5,6,7::5,2,3,7,1,0,6,4::1,2,3,4,0,6,7,5::1,0,2,3,4,5,6,7::0,7,1,3,4,5,6,2::5,3,6,4,1,0,2,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# df5a92a77db899799b71ec75d1970343
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=5657 --custom_alignment=0,1,2,5,4,3,6,7::0,2,5,3,1,4,6,7::1,2,3,0,4,5,6,7::5,2,3,7,1,0,6,4::1,2,3,4,0,6,7,5::1,0,2,3,4,5,6,7::0,7,1,3,4,5,6,2::5,3,6,4,1,0,2,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,1,2,7,3,6,5,4::1,5,0,3,4,7,6,2::1,2,6,0,4,5,3,7::2,6,4,3,5,0,1,7::1,0,3,4,6,5,2,7::1,3,0,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7

# c60c44dc4a5d8c5b02eea77fd33e85cc
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1508 --custom_alignment=0,1,2,7,3,6,5,4::1,5,0,3,4,7,6,2::1,2,6,0,4,5,3,7::2,6,4,3,5,0,1,7::1,0,3,4,6,5,2,7::1,3,0,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c60c44dc4a5d8c5b02eea77fd33e85cc
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=6728 --custom_alignment=0,1,2,7,3,6,5,4::1,5,0,3,4,7,6,2::1,2,6,0,4,5,3,7::2,6,4,3,5,0,1,7::1,0,3,4,6,5,2,7::1,3,0,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c60c44dc4a5d8c5b02eea77fd33e85cc
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=5667 --custom_alignment=0,1,2,7,3,6,5,4::1,5,0,3,4,7,6,2::1,2,6,0,4,5,3,7::2,6,4,3,5,0,1,7::1,0,3,4,6,5,2,7::1,3,0,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c60c44dc4a5d8c5b02eea77fd33e85cc
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=7967 --custom_alignment=0,1,2,7,3,6,5,4::1,5,0,3,4,7,6,2::1,2,6,0,4,5,3,7::2,6,4,3,5,0,1,7::1,0,3,4,6,5,2,7::1,3,0,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c60c44dc4a5d8c5b02eea77fd33e85cc
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1125 --custom_alignment=0,1,2,7,3,6,5,4::1,5,0,3,4,7,6,2::1,2,6,0,4,5,3,7::2,6,4,3,5,0,1,7::1,0,3,4,6,5,2,7::1,3,0,2,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 2,1,0,5,4,3,6,7::2,1,4,3,0,5,6,7::1,2,3,0,4,5,7,6::0,2,3,4,5,1,6,7::4,1,3,2,5,6,0,7::1,0,2,3,4,5,6,7::1,0,3,6,2,4,5,7::2,3,0,1,4,5,6,7

# 7c144b9599d1e1308c5c4090c0b83df7
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=5212 --custom_alignment=2,1,0,5,4,3,6,7::2,1,4,3,0,5,6,7::1,2,3,0,4,5,7,6::0,2,3,4,5,1,6,7::4,1,3,2,5,6,0,7::1,0,2,3,4,5,6,7::1,0,3,6,2,4,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 7c144b9599d1e1308c5c4090c0b83df7
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8885 --custom_alignment=2,1,0,5,4,3,6,7::2,1,4,3,0,5,6,7::1,2,3,0,4,5,7,6::0,2,3,4,5,1,6,7::4,1,3,2,5,6,0,7::1,0,2,3,4,5,6,7::1,0,3,6,2,4,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 7c144b9599d1e1308c5c4090c0b83df7
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3835 --custom_alignment=2,1,0,5,4,3,6,7::2,1,4,3,0,5,6,7::1,2,3,0,4,5,7,6::0,2,3,4,5,1,6,7::4,1,3,2,5,6,0,7::1,0,2,3,4,5,6,7::1,0,3,6,2,4,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 7c144b9599d1e1308c5c4090c0b83df7
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=5153 --custom_alignment=2,1,0,5,4,3,6,7::2,1,4,3,0,5,6,7::1,2,3,0,4,5,7,6::0,2,3,4,5,1,6,7::4,1,3,2,5,6,0,7::1,0,2,3,4,5,6,7::1,0,3,6,2,4,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 7c144b9599d1e1308c5c4090c0b83df7
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1405 --custom_alignment=2,1,0,5,4,3,6,7::2,1,4,3,0,5,6,7::1,2,3,0,4,5,7,6::0,2,3,4,5,1,6,7::4,1,3,2,5,6,0,7::1,0,2,3,4,5,6,7::1,0,3,6,2,4,5,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,1,2,6,5,4,7,3::4,2,1,5,6,3,0,7::1,6,3,0,4,5,2,7::1,2,4,3,5,0,6,7::1,0,3,5,4,6,2,7::6,1,2,3,4,5,0,7::2,0,1,3,4,5,6,7::4,3,0,1,2,5,6,7

# 97abf465da7ee3369abf35be72f60579
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3272 --custom_alignment=0,1,2,6,5,4,7,3::4,2,1,5,6,3,0,7::1,6,3,0,4,5,2,7::1,2,4,3,5,0,6,7::1,0,3,5,4,6,2,7::6,1,2,3,4,5,0,7::2,0,1,3,4,5,6,7::4,3,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 97abf465da7ee3369abf35be72f60579
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=6804 --custom_alignment=0,1,2,6,5,4,7,3::4,2,1,5,6,3,0,7::1,6,3,0,4,5,2,7::1,2,4,3,5,0,6,7::1,0,3,5,4,6,2,7::6,1,2,3,4,5,0,7::2,0,1,3,4,5,6,7::4,3,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 97abf465da7ee3369abf35be72f60579
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1303 --custom_alignment=0,1,2,6,5,4,7,3::4,2,1,5,6,3,0,7::1,6,3,0,4,5,2,7::1,2,4,3,5,0,6,7::1,0,3,5,4,6,2,7::6,1,2,3,4,5,0,7::2,0,1,3,4,5,6,7::4,3,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 97abf465da7ee3369abf35be72f60579
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8984 --custom_alignment=0,1,2,6,5,4,7,3::4,2,1,5,6,3,0,7::1,6,3,0,4,5,2,7::1,2,4,3,5,0,6,7::1,0,3,5,4,6,2,7::6,1,2,3,4,5,0,7::2,0,1,3,4,5,6,7::4,3,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 97abf465da7ee3369abf35be72f60579
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=892 --custom_alignment=0,1,2,6,5,4,7,3::4,2,1,5,6,3,0,7::1,6,3,0,4,5,2,7::1,2,4,3,5,0,6,7::1,0,3,5,4,6,2,7::6,1,2,3,4,5,0,7::2,0,1,3,4,5,6,7::4,3,0,1,2,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 4,1,0,6,2,5,3,7::2,5,1,3,4,0,6,7::1,2,0,3,4,5,6,7::1,5,3,4,2,0,6,7::7,2,3,1,5,6,0,4::1,3,2,4,0,7,6,5::2,0,1,3,4,5,6,7::2,3,0,6,5,7,1,4

# db7306ab94f4aac747be99dee24f7dbb
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4211 --custom_alignment=4,1,0,6,2,5,3,7::2,5,1,3,4,0,6,7::1,2,0,3,4,5,6,7::1,5,3,4,2,0,6,7::7,2,3,1,5,6,0,4::1,3,2,4,0,7,6,5::2,0,1,3,4,5,6,7::2,3,0,6,5,7,1,4 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# db7306ab94f4aac747be99dee24f7dbb
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=6491 --custom_alignment=4,1,0,6,2,5,3,7::2,5,1,3,4,0,6,7::1,2,0,3,4,5,6,7::1,5,3,4,2,0,6,7::7,2,3,1,5,6,0,4::1,3,2,4,0,7,6,5::2,0,1,3,4,5,6,7::2,3,0,6,5,7,1,4 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# db7306ab94f4aac747be99dee24f7dbb
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8231 --custom_alignment=4,1,0,6,2,5,3,7::2,5,1,3,4,0,6,7::1,2,0,3,4,5,6,7::1,5,3,4,2,0,6,7::7,2,3,1,5,6,0,4::1,3,2,4,0,7,6,5::2,0,1,3,4,5,6,7::2,3,0,6,5,7,1,4 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# db7306ab94f4aac747be99dee24f7dbb
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8063 --custom_alignment=4,1,0,6,2,5,3,7::2,5,1,3,4,0,6,7::1,2,0,3,4,5,6,7::1,5,3,4,2,0,6,7::7,2,3,1,5,6,0,4::1,3,2,4,0,7,6,5::2,0,1,3,4,5,6,7::2,3,0,6,5,7,1,4 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# db7306ab94f4aac747be99dee24f7dbb
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3995 --custom_alignment=4,1,0,6,2,5,3,7::2,5,1,3,4,0,6,7::1,2,0,3,4,5,6,7::1,5,3,4,2,0,6,7::7,2,3,1,5,6,0,4::1,3,2,4,0,7,6,5::2,0,1,3,4,5,6,7::2,3,0,6,5,7,1,4 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 7,1,2,3,4,5,6,0::6,2,1,3,4,0,5,7::1,0,3,2,4,5,6,7::1,3,6,2,4,0,5,7::1,2,3,4,7,6,5,0::4,3,2,5,1,0,6,7::5,0,6,3,4,2,1,7::2,3,0,1,4,5,6,7

# 58d804231b4fef8dff98232e1903f8e9
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=7929 --custom_alignment=7,1,2,3,4,5,6,0::6,2,1,3,4,0,5,7::1,0,3,2,4,5,6,7::1,3,6,2,4,0,5,7::1,2,3,4,7,6,5,0::4,3,2,5,1,0,6,7::5,0,6,3,4,2,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 58d804231b4fef8dff98232e1903f8e9
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8742 --custom_alignment=7,1,2,3,4,5,6,0::6,2,1,3,4,0,5,7::1,0,3,2,4,5,6,7::1,3,6,2,4,0,5,7::1,2,3,4,7,6,5,0::4,3,2,5,1,0,6,7::5,0,6,3,4,2,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 58d804231b4fef8dff98232e1903f8e9
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=7351 --custom_alignment=7,1,2,3,4,5,6,0::6,2,1,3,4,0,5,7::1,0,3,2,4,5,6,7::1,3,6,2,4,0,5,7::1,2,3,4,7,6,5,0::4,3,2,5,1,0,6,7::5,0,6,3,4,2,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 58d804231b4fef8dff98232e1903f8e9
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9129 --custom_alignment=7,1,2,3,4,5,6,0::6,2,1,3,4,0,5,7::1,0,3,2,4,5,6,7::1,3,6,2,4,0,5,7::1,2,3,4,7,6,5,0::4,3,2,5,1,0,6,7::5,0,6,3,4,2,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 58d804231b4fef8dff98232e1903f8e9
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=7476 --custom_alignment=7,1,2,3,4,5,6,0::6,2,1,3,4,0,5,7::1,0,3,2,4,5,6,7::1,3,6,2,4,0,5,7::1,2,3,4,7,6,5,0::4,3,2,5,1,0,6,7::5,0,6,3,4,2,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,1,4,3,5,2,6,7::1,2,0,5,4,3,6,7::4,0,1,7,3,2,5,6::4,2,3,1,5,0,6,7::1,3,4,0,5,6,2,7::1,0,2,3,4,5,6,7::4,0,1,2,5,3,6,7::2,3,0,1,4,5,6,7

# 55a717785a52d2bc80dd740b75a8de8a
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=5793 --custom_alignment=0,1,4,3,5,2,6,7::1,2,0,5,4,3,6,7::4,0,1,7,3,2,5,6::4,2,3,1,5,0,6,7::1,3,4,0,5,6,2,7::1,0,2,3,4,5,6,7::4,0,1,2,5,3,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 55a717785a52d2bc80dd740b75a8de8a
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=7411 --custom_alignment=0,1,4,3,5,2,6,7::1,2,0,5,4,3,6,7::4,0,1,7,3,2,5,6::4,2,3,1,5,0,6,7::1,3,4,0,5,6,2,7::1,0,2,3,4,5,6,7::4,0,1,2,5,3,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 55a717785a52d2bc80dd740b75a8de8a
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9067 --custom_alignment=0,1,4,3,5,2,6,7::1,2,0,5,4,3,6,7::4,0,1,7,3,2,5,6::4,2,3,1,5,0,6,7::1,3,4,0,5,6,2,7::1,0,2,3,4,5,6,7::4,0,1,2,5,3,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 55a717785a52d2bc80dd740b75a8de8a
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3579 --custom_alignment=0,1,4,3,5,2,6,7::1,2,0,5,4,3,6,7::4,0,1,7,3,2,5,6::4,2,3,1,5,0,6,7::1,3,4,0,5,6,2,7::1,0,2,3,4,5,6,7::4,0,1,2,5,3,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 55a717785a52d2bc80dd740b75a8de8a
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=683 --custom_alignment=0,1,4,3,5,2,6,7::1,2,0,5,4,3,6,7::4,0,1,7,3,2,5,6::4,2,3,1,5,0,6,7::1,3,4,0,5,6,2,7::1,0,2,3,4,5,6,7::4,0,1,2,5,3,6,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,1,2,3,7,5,6,4::1,7,0,3,6,5,4,2::6,2,3,7,4,1,5,0::1,2,3,4,5,0,7,6::2,1,3,4,7,6,0,5::1,3,2,0,4,5,6,7::2,0,5,3,4,1,6,7::0,3,2,1,4,5,6,7

# b2479647466965ca03ff1c99ef294549
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=5282 --custom_alignment=0,1,2,3,7,5,6,4::1,7,0,3,6,5,4,2::6,2,3,7,4,1,5,0::1,2,3,4,5,0,7,6::2,1,3,4,7,6,0,5::1,3,2,0,4,5,6,7::2,0,5,3,4,1,6,7::0,3,2,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b2479647466965ca03ff1c99ef294549
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=2973 --custom_alignment=0,1,2,3,7,5,6,4::1,7,0,3,6,5,4,2::6,2,3,7,4,1,5,0::1,2,3,4,5,0,7,6::2,1,3,4,7,6,0,5::1,3,2,0,4,5,6,7::2,0,5,3,4,1,6,7::0,3,2,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b2479647466965ca03ff1c99ef294549
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9319 --custom_alignment=0,1,2,3,7,5,6,4::1,7,0,3,6,5,4,2::6,2,3,7,4,1,5,0::1,2,3,4,5,0,7,6::2,1,3,4,7,6,0,5::1,3,2,0,4,5,6,7::2,0,5,3,4,1,6,7::0,3,2,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b2479647466965ca03ff1c99ef294549
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9318 --custom_alignment=0,1,2,3,7,5,6,4::1,7,0,3,6,5,4,2::6,2,3,7,4,1,5,0::1,2,3,4,5,0,7,6::2,1,3,4,7,6,0,5::1,3,2,0,4,5,6,7::2,0,5,3,4,1,6,7::0,3,2,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b2479647466965ca03ff1c99ef294549
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=895 --custom_alignment=0,1,2,3,7,5,6,4::1,7,0,3,6,5,4,2::6,2,3,7,4,1,5,0::1,2,3,4,5,0,7,6::2,1,3,4,7,6,0,5::1,3,2,0,4,5,6,7::2,0,5,3,4,1,6,7::0,3,2,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,1,2,4,3,5,6,7::1,0,2,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,7,6,0::1,2,7,4,6,5,0,3::5,4,3,2,0,6,1,7::7,0,1,3,4,5,6,2::2,3,7,1,4,0,5,6

# a8bf82aeb89b18a291a61d280b6d3d21
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9980 --custom_alignment=0,1,2,4,3,5,6,7::1,0,2,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,7,6,0::1,2,7,4,6,5,0,3::5,4,3,2,0,6,1,7::7,0,1,3,4,5,6,2::2,3,7,1,4,0,5,6 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a8bf82aeb89b18a291a61d280b6d3d21
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=878 --custom_alignment=0,1,2,4,3,5,6,7::1,0,2,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,7,6,0::1,2,7,4,6,5,0,3::5,4,3,2,0,6,1,7::7,0,1,3,4,5,6,2::2,3,7,1,4,0,5,6 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a8bf82aeb89b18a291a61d280b6d3d21
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=2989 --custom_alignment=0,1,2,4,3,5,6,7::1,0,2,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,7,6,0::1,2,7,4,6,5,0,3::5,4,3,2,0,6,1,7::7,0,1,3,4,5,6,2::2,3,7,1,4,0,5,6 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a8bf82aeb89b18a291a61d280b6d3d21
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=5975 --custom_alignment=0,1,2,4,3,5,6,7::1,0,2,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,7,6,0::1,2,7,4,6,5,0,3::5,4,3,2,0,6,1,7::7,0,1,3,4,5,6,2::2,3,7,1,4,0,5,6 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a8bf82aeb89b18a291a61d280b6d3d21
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4497 --custom_alignment=0,1,2,4,3,5,6,7::1,0,2,3,4,5,6,7::1,2,3,0,4,5,6,7::3,2,1,4,5,7,6,0::1,2,7,4,6,5,0,3::5,4,3,2,0,6,1,7::7,0,1,3,4,5,6,2::2,3,7,1,4,0,5,6 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,4,2,3,7,1,6,5::6,7,1,3,0,5,4,2::1,2,3,0,4,6,5,7::1,2,3,4,5,0,6,7::6,2,3,5,4,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7

# b0e8718d0f92fadb7a6ad9ea0daa1353
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9006 --custom_alignment=0,4,2,3,7,1,6,5::6,7,1,3,0,5,4,2::1,2,3,0,4,6,5,7::1,2,3,4,5,0,6,7::6,2,3,5,4,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b0e8718d0f92fadb7a6ad9ea0daa1353
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=2480 --custom_alignment=0,4,2,3,7,1,6,5::6,7,1,3,0,5,4,2::1,2,3,0,4,6,5,7::1,2,3,4,5,0,6,7::6,2,3,5,4,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b0e8718d0f92fadb7a6ad9ea0daa1353
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3644 --custom_alignment=0,4,2,3,7,1,6,5::6,7,1,3,0,5,4,2::1,2,3,0,4,6,5,7::1,2,3,4,5,0,6,7::6,2,3,5,4,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b0e8718d0f92fadb7a6ad9ea0daa1353
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4253 --custom_alignment=0,4,2,3,7,1,6,5::6,7,1,3,0,5,4,2::1,2,3,0,4,6,5,7::1,2,3,4,5,0,6,7::6,2,3,5,4,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b0e8718d0f92fadb7a6ad9ea0daa1353
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4877 --custom_alignment=0,4,2,3,7,1,6,5::6,7,1,3,0,5,4,2::1,2,3,0,4,6,5,7::1,2,3,4,5,0,6,7::6,2,3,5,4,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,1,2,3,4,5,6,7::6,1,0,2,4,5,3,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,0,4,3,1,5,7::1,6,2,3,4,5,0,7::2,0,1,6,4,7,5,3::2,3,0,1,4,5,6,7

# c3ef005cd7b44e114050f60aedcd3ac8
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8355 --custom_alignment=0,1,2,3,4,5,6,7::6,1,0,2,4,5,3,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,0,4,3,1,5,7::1,6,2,3,4,5,0,7::2,0,1,6,4,7,5,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c3ef005cd7b44e114050f60aedcd3ac8
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4394 --custom_alignment=0,1,2,3,4,5,6,7::6,1,0,2,4,5,3,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,0,4,3,1,5,7::1,6,2,3,4,5,0,7::2,0,1,6,4,7,5,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c3ef005cd7b44e114050f60aedcd3ac8
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=2594 --custom_alignment=0,1,2,3,4,5,6,7::6,1,0,2,4,5,3,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,0,4,3,1,5,7::1,6,2,3,4,5,0,7::2,0,1,6,4,7,5,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c3ef005cd7b44e114050f60aedcd3ac8
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=7284 --custom_alignment=0,1,2,3,4,5,6,7::6,1,0,2,4,5,3,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,0,4,3,1,5,7::1,6,2,3,4,5,0,7::2,0,1,6,4,7,5,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c3ef005cd7b44e114050f60aedcd3ac8
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=8990 --custom_alignment=0,1,2,3,4,5,6,7::6,1,0,2,4,5,3,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,0,4,3,1,5,7::1,6,2,3,4,5,0,7::2,0,1,6,4,7,5,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 2,1,0,5,3,4,6,7::1,5,0,2,4,3,6,7::5,2,6,0,4,1,3,7::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::2,3,1,0,4,5,6,7::2,3,0,7,4,5,6,1

# b32064174ecd66db18ddfc352144afd1
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3664 --custom_alignment=2,1,0,5,3,4,6,7::1,5,0,2,4,3,6,7::5,2,6,0,4,1,3,7::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::2,3,1,0,4,5,6,7::2,3,0,7,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b32064174ecd66db18ddfc352144afd1
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=4218 --custom_alignment=2,1,0,5,3,4,6,7::1,5,0,2,4,3,6,7::5,2,6,0,4,1,3,7::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::2,3,1,0,4,5,6,7::2,3,0,7,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b32064174ecd66db18ddfc352144afd1
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=56 --custom_alignment=2,1,0,5,3,4,6,7::1,5,0,2,4,3,6,7::5,2,6,0,4,1,3,7::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::2,3,1,0,4,5,6,7::2,3,0,7,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b32064174ecd66db18ddfc352144afd1
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1231 --custom_alignment=2,1,0,5,3,4,6,7::1,5,0,2,4,3,6,7::5,2,6,0,4,1,3,7::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::2,3,1,0,4,5,6,7::2,3,0,7,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# b32064174ecd66db18ddfc352144afd1
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=3155 --custom_alignment=2,1,0,5,3,4,6,7::1,5,0,2,4,3,6,7::5,2,6,0,4,1,3,7::1,2,5,4,3,0,6,7::1,2,3,4,5,6,0,7::1,0,5,3,4,2,6,7::2,3,1,0,4,5,6,7::2,3,0,7,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 16 alignment for 9xx: 0,7,2,5,4,3,6,1::2,1,0,3,4,5,6,7::4,3,2,0,6,1,7,5::1,4,3,2,5,0,6,7::1,2,3,4,6,5,0,7::1,0,6,3,4,5,2,7::2,4,1,3,7,5,6,0::2,4,0,1,3,5,6,7

# e0cc6709ca1e063bdeee3a2155abbafc
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=1062 --custom_alignment=0,7,2,5,4,3,6,1::2,1,0,3,4,5,6,7::4,3,2,0,6,1,7,5::1,4,3,2,5,0,6,7::1,2,3,4,6,5,0,7::1,0,6,3,4,5,2,7::2,4,1,3,7,5,6,0::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# e0cc6709ca1e063bdeee3a2155abbafc
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=9116 --custom_alignment=0,7,2,5,4,3,6,1::2,1,0,3,4,5,6,7::4,3,2,0,6,1,7,5::1,4,3,2,5,0,6,7::1,2,3,4,6,5,0,7::1,0,6,3,4,5,2,7::2,4,1,3,7,5,6,0::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# e0cc6709ca1e063bdeee3a2155abbafc
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=885 --custom_alignment=0,7,2,5,4,3,6,1::2,1,0,3,4,5,6,7::4,3,2,0,6,1,7,5::1,4,3,2,5,0,6,7::1,2,3,4,6,5,0,7::1,0,6,3,4,5,2,7::2,4,1,3,7,5,6,0::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# e0cc6709ca1e063bdeee3a2155abbafc
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=6176 --custom_alignment=0,7,2,5,4,3,6,1::2,1,0,3,4,5,6,7::4,3,2,0,6,1,7,5::1,4,3,2,5,0,6,7::1,2,3,4,6,5,0,7::1,0,6,3,4,5,2,7::2,4,1,3,7,5,6,0::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# e0cc6709ca1e063bdeee3a2155abbafc
sbatch -J search_9xx_mutate_16 submit.sh python 1.train.py --seed=75 --custom_alignment=0,7,2,5,4,3,6,1::2,1,0,3,4,5,6,7::4,3,2,0,6,1,7,5::1,4,3,2,5,0,6,7::1,2,3,4,6,5,0,7::1,0,6,3,4,5,2,7::2,4,1,3,7,5,6,0::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_16 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 7,1,4,3,0,5,6,2::2,1,7,3,4,5,0,6::4,2,3,0,7,5,6,1::1,2,4,0,5,3,6,7::1,2,4,7,5,6,0,3::1,6,2,7,4,5,3,0::3,1,0,6,4,5,2,7::2,7,0,5,4,1,6,3

# a041bdd81612a23419b7ec11ecc7416d
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8830 --custom_alignment=7,1,4,3,0,5,6,2::2,1,7,3,4,5,0,6::4,2,3,0,7,5,6,1::1,2,4,0,5,3,6,7::1,2,4,7,5,6,0,3::1,6,2,7,4,5,3,0::3,1,0,6,4,5,2,7::2,7,0,5,4,1,6,3 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a041bdd81612a23419b7ec11ecc7416d
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=7367 --custom_alignment=7,1,4,3,0,5,6,2::2,1,7,3,4,5,0,6::4,2,3,0,7,5,6,1::1,2,4,0,5,3,6,7::1,2,4,7,5,6,0,3::1,6,2,7,4,5,3,0::3,1,0,6,4,5,2,7::2,7,0,5,4,1,6,3 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a041bdd81612a23419b7ec11ecc7416d
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1676 --custom_alignment=7,1,4,3,0,5,6,2::2,1,7,3,4,5,0,6::4,2,3,0,7,5,6,1::1,2,4,0,5,3,6,7::1,2,4,7,5,6,0,3::1,6,2,7,4,5,3,0::3,1,0,6,4,5,2,7::2,7,0,5,4,1,6,3 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a041bdd81612a23419b7ec11ecc7416d
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1000 --custom_alignment=7,1,4,3,0,5,6,2::2,1,7,3,4,5,0,6::4,2,3,0,7,5,6,1::1,2,4,0,5,3,6,7::1,2,4,7,5,6,0,3::1,6,2,7,4,5,3,0::3,1,0,6,4,5,2,7::2,7,0,5,4,1,6,3 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a041bdd81612a23419b7ec11ecc7416d
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=3632 --custom_alignment=7,1,4,3,0,5,6,2::2,1,7,3,4,5,0,6::4,2,3,0,7,5,6,1::1,2,4,0,5,3,6,7::1,2,4,7,5,6,0,3::1,6,2,7,4,5,3,0::3,1,0,6,4,5,2,7::2,7,0,5,4,1,6,3 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 6,4,2,3,1,5,0,7::1,5,0,7,4,3,6,2::4,5,7,2,1,0,6,3::0,2,3,4,5,7,6,1::1,5,7,2,0,6,3,4::4,2,6,3,1,5,7,0::6,5,4,3,1,0,2,7::2,5,0,1,4,3,6,7

# 47c150d68c71528e1c013be83f2a8eb9
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=6088 --custom_alignment=6,4,2,3,1,5,0,7::1,5,0,7,4,3,6,2::4,5,7,2,1,0,6,3::0,2,3,4,5,7,6,1::1,5,7,2,0,6,3,4::4,2,6,3,1,5,7,0::6,5,4,3,1,0,2,7::2,5,0,1,4,3,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 47c150d68c71528e1c013be83f2a8eb9
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=6936 --custom_alignment=6,4,2,3,1,5,0,7::1,5,0,7,4,3,6,2::4,5,7,2,1,0,6,3::0,2,3,4,5,7,6,1::1,5,7,2,0,6,3,4::4,2,6,3,1,5,7,0::6,5,4,3,1,0,2,7::2,5,0,1,4,3,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 47c150d68c71528e1c013be83f2a8eb9
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=4224 --custom_alignment=6,4,2,3,1,5,0,7::1,5,0,7,4,3,6,2::4,5,7,2,1,0,6,3::0,2,3,4,5,7,6,1::1,5,7,2,0,6,3,4::4,2,6,3,1,5,7,0::6,5,4,3,1,0,2,7::2,5,0,1,4,3,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 47c150d68c71528e1c013be83f2a8eb9
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=4589 --custom_alignment=6,4,2,3,1,5,0,7::1,5,0,7,4,3,6,2::4,5,7,2,1,0,6,3::0,2,3,4,5,7,6,1::1,5,7,2,0,6,3,4::4,2,6,3,1,5,7,0::6,5,4,3,1,0,2,7::2,5,0,1,4,3,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 47c150d68c71528e1c013be83f2a8eb9
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9783 --custom_alignment=6,4,2,3,1,5,0,7::1,5,0,7,4,3,6,2::4,5,7,2,1,0,6,3::0,2,3,4,5,7,6,1::1,5,7,2,0,6,3,4::4,2,6,3,1,5,7,0::6,5,4,3,1,0,2,7::2,5,0,1,4,3,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 4,2,1,3,5,0,6,7::1,2,7,3,4,5,6,0::7,0,4,2,3,5,6,1::5,2,6,4,1,0,7,3::4,6,1,3,5,0,2,7::6,0,2,7,4,5,1,3::3,6,1,2,0,5,4,7::3,4,2,1,0,7,6,5

# c9fa79d6ef6f68abf0481758a981d73e
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9200 --custom_alignment=4,2,1,3,5,0,6,7::1,2,7,3,4,5,6,0::7,0,4,2,3,5,6,1::5,2,6,4,1,0,7,3::4,6,1,3,5,0,2,7::6,0,2,7,4,5,1,3::3,6,1,2,0,5,4,7::3,4,2,1,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c9fa79d6ef6f68abf0481758a981d73e
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8939 --custom_alignment=4,2,1,3,5,0,6,7::1,2,7,3,4,5,6,0::7,0,4,2,3,5,6,1::5,2,6,4,1,0,7,3::4,6,1,3,5,0,2,7::6,0,2,7,4,5,1,3::3,6,1,2,0,5,4,7::3,4,2,1,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c9fa79d6ef6f68abf0481758a981d73e
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9452 --custom_alignment=4,2,1,3,5,0,6,7::1,2,7,3,4,5,6,0::7,0,4,2,3,5,6,1::5,2,6,4,1,0,7,3::4,6,1,3,5,0,2,7::6,0,2,7,4,5,1,3::3,6,1,2,0,5,4,7::3,4,2,1,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c9fa79d6ef6f68abf0481758a981d73e
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5691 --custom_alignment=4,2,1,3,5,0,6,7::1,2,7,3,4,5,6,0::7,0,4,2,3,5,6,1::5,2,6,4,1,0,7,3::4,6,1,3,5,0,2,7::6,0,2,7,4,5,1,3::3,6,1,2,0,5,4,7::3,4,2,1,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c9fa79d6ef6f68abf0481758a981d73e
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=6405 --custom_alignment=4,2,1,3,5,0,6,7::1,2,7,3,4,5,6,0::7,0,4,2,3,5,6,1::5,2,6,4,1,0,7,3::4,6,1,3,5,0,2,7::6,0,2,7,4,5,1,3::3,6,1,2,0,5,4,7::3,4,2,1,0,7,6,5 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 2,1,5,0,4,3,7,6::5,2,0,3,6,7,4,1::0,4,1,3,7,2,6,5::7,2,3,4,5,0,6,1::1,2,3,4,5,7,0,6::6,1,2,0,4,3,5,7::7,5,4,3,1,0,6,2::3,2,1,0,4,5,6,7

# 9e4b6b865444a96174c92c59be64ae1f
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2706 --custom_alignment=2,1,5,0,4,3,7,6::5,2,0,3,6,7,4,1::0,4,1,3,7,2,6,5::7,2,3,4,5,0,6,1::1,2,3,4,5,7,0,6::6,1,2,0,4,3,5,7::7,5,4,3,1,0,6,2::3,2,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9e4b6b865444a96174c92c59be64ae1f
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=6031 --custom_alignment=2,1,5,0,4,3,7,6::5,2,0,3,6,7,4,1::0,4,1,3,7,2,6,5::7,2,3,4,5,0,6,1::1,2,3,4,5,7,0,6::6,1,2,0,4,3,5,7::7,5,4,3,1,0,6,2::3,2,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9e4b6b865444a96174c92c59be64ae1f
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=6105 --custom_alignment=2,1,5,0,4,3,7,6::5,2,0,3,6,7,4,1::0,4,1,3,7,2,6,5::7,2,3,4,5,0,6,1::1,2,3,4,5,7,0,6::6,1,2,0,4,3,5,7::7,5,4,3,1,0,6,2::3,2,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9e4b6b865444a96174c92c59be64ae1f
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=3844 --custom_alignment=2,1,5,0,4,3,7,6::5,2,0,3,6,7,4,1::0,4,1,3,7,2,6,5::7,2,3,4,5,0,6,1::1,2,3,4,5,7,0,6::6,1,2,0,4,3,5,7::7,5,4,3,1,0,6,2::3,2,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9e4b6b865444a96174c92c59be64ae1f
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=352 --custom_alignment=2,1,5,0,4,3,7,6::5,2,0,3,6,7,4,1::0,4,1,3,7,2,6,5::7,2,3,4,5,0,6,1::1,2,3,4,5,7,0,6::6,1,2,0,4,3,5,7::7,5,4,3,1,0,6,2::3,2,1,0,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 3,0,7,1,4,5,6,2::1,2,0,3,4,6,5,7::6,2,7,0,4,5,3,1::1,2,3,4,5,7,6,0::4,2,0,5,1,6,3,7::3,0,7,1,4,2,5,6::6,7,1,3,4,5,2,0::3,2,7,6,4,5,0,1

# 24a9ed442ca6ca40a37d9d9f9ebb8482
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=7083 --custom_alignment=3,0,7,1,4,5,6,2::1,2,0,3,4,6,5,7::6,2,7,0,4,5,3,1::1,2,3,4,5,7,6,0::4,2,0,5,1,6,3,7::3,0,7,1,4,2,5,6::6,7,1,3,4,5,2,0::3,2,7,6,4,5,0,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24a9ed442ca6ca40a37d9d9f9ebb8482
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2229 --custom_alignment=3,0,7,1,4,5,6,2::1,2,0,3,4,6,5,7::6,2,7,0,4,5,3,1::1,2,3,4,5,7,6,0::4,2,0,5,1,6,3,7::3,0,7,1,4,2,5,6::6,7,1,3,4,5,2,0::3,2,7,6,4,5,0,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24a9ed442ca6ca40a37d9d9f9ebb8482
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=4562 --custom_alignment=3,0,7,1,4,5,6,2::1,2,0,3,4,6,5,7::6,2,7,0,4,5,3,1::1,2,3,4,5,7,6,0::4,2,0,5,1,6,3,7::3,0,7,1,4,2,5,6::6,7,1,3,4,5,2,0::3,2,7,6,4,5,0,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24a9ed442ca6ca40a37d9d9f9ebb8482
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=7601 --custom_alignment=3,0,7,1,4,5,6,2::1,2,0,3,4,6,5,7::6,2,7,0,4,5,3,1::1,2,3,4,5,7,6,0::4,2,0,5,1,6,3,7::3,0,7,1,4,2,5,6::6,7,1,3,4,5,2,0::3,2,7,6,4,5,0,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 24a9ed442ca6ca40a37d9d9f9ebb8482
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8002 --custom_alignment=3,0,7,1,4,5,6,2::1,2,0,3,4,6,5,7::6,2,7,0,4,5,3,1::1,2,3,4,5,7,6,0::4,2,0,5,1,6,3,7::3,0,7,1,4,2,5,6::6,7,1,3,4,5,2,0::3,2,7,6,4,5,0,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 6,7,3,2,0,5,4,1::7,2,5,3,4,0,6,1::1,2,3,0,4,5,6,7::1,3,7,4,5,2,6,0::1,4,3,5,7,6,2,0::1,5,2,3,4,0,6,7::7,3,1,0,4,5,6,2::2,5,3,4,1,0,6,7

# f721fae18f17c24b3a9496263a14d839
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1722 --custom_alignment=6,7,3,2,0,5,4,1::7,2,5,3,4,0,6,1::1,2,3,0,4,5,6,7::1,3,7,4,5,2,6,0::1,4,3,5,7,6,2,0::1,5,2,3,4,0,6,7::7,3,1,0,4,5,6,2::2,5,3,4,1,0,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f721fae18f17c24b3a9496263a14d839
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9619 --custom_alignment=6,7,3,2,0,5,4,1::7,2,5,3,4,0,6,1::1,2,3,0,4,5,6,7::1,3,7,4,5,2,6,0::1,4,3,5,7,6,2,0::1,5,2,3,4,0,6,7::7,3,1,0,4,5,6,2::2,5,3,4,1,0,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f721fae18f17c24b3a9496263a14d839
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8362 --custom_alignment=6,7,3,2,0,5,4,1::7,2,5,3,4,0,6,1::1,2,3,0,4,5,6,7::1,3,7,4,5,2,6,0::1,4,3,5,7,6,2,0::1,5,2,3,4,0,6,7::7,3,1,0,4,5,6,2::2,5,3,4,1,0,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f721fae18f17c24b3a9496263a14d839
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=7334 --custom_alignment=6,7,3,2,0,5,4,1::7,2,5,3,4,0,6,1::1,2,3,0,4,5,6,7::1,3,7,4,5,2,6,0::1,4,3,5,7,6,2,0::1,5,2,3,4,0,6,7::7,3,1,0,4,5,6,2::2,5,3,4,1,0,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f721fae18f17c24b3a9496263a14d839
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5162 --custom_alignment=6,7,3,2,0,5,4,1::7,2,5,3,4,0,6,1::1,2,3,0,4,5,6,7::1,3,7,4,5,2,6,0::1,4,3,5,7,6,2,0::1,5,2,3,4,0,6,7::7,3,1,0,4,5,6,2::2,5,3,4,1,0,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 0,1,2,5,4,3,6,7::1,0,5,3,4,2,6,7::1,3,0,6,5,4,2,7::1,5,6,0,2,4,3,7::6,0,3,1,7,2,4,5::7,0,2,3,4,6,1,5::2,0,3,1,4,5,6,7::2,3,4,1,0,5,6,7

# a9f139ef752d5723a80fab0c6bc2dad4
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2335 --custom_alignment=0,1,2,5,4,3,6,7::1,0,5,3,4,2,6,7::1,3,0,6,5,4,2,7::1,5,6,0,2,4,3,7::6,0,3,1,7,2,4,5::7,0,2,3,4,6,1,5::2,0,3,1,4,5,6,7::2,3,4,1,0,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a9f139ef752d5723a80fab0c6bc2dad4
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8409 --custom_alignment=0,1,2,5,4,3,6,7::1,0,5,3,4,2,6,7::1,3,0,6,5,4,2,7::1,5,6,0,2,4,3,7::6,0,3,1,7,2,4,5::7,0,2,3,4,6,1,5::2,0,3,1,4,5,6,7::2,3,4,1,0,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a9f139ef752d5723a80fab0c6bc2dad4
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=4136 --custom_alignment=0,1,2,5,4,3,6,7::1,0,5,3,4,2,6,7::1,3,0,6,5,4,2,7::1,5,6,0,2,4,3,7::6,0,3,1,7,2,4,5::7,0,2,3,4,6,1,5::2,0,3,1,4,5,6,7::2,3,4,1,0,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a9f139ef752d5723a80fab0c6bc2dad4
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=804 --custom_alignment=0,1,2,5,4,3,6,7::1,0,5,3,4,2,6,7::1,3,0,6,5,4,2,7::1,5,6,0,2,4,3,7::6,0,3,1,7,2,4,5::7,0,2,3,4,6,1,5::2,0,3,1,4,5,6,7::2,3,4,1,0,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a9f139ef752d5723a80fab0c6bc2dad4
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=443 --custom_alignment=0,1,2,5,4,3,6,7::1,0,5,3,4,2,6,7::1,3,0,6,5,4,2,7::1,5,6,0,2,4,3,7::6,0,3,1,7,2,4,5::7,0,2,3,4,6,1,5::2,0,3,1,4,5,6,7::2,3,4,1,0,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 1,0,2,3,7,5,6,4::7,1,0,3,4,5,6,2::1,2,3,5,0,4,6,7::3,6,1,2,5,4,0,7::6,7,3,2,0,4,5,1::1,0,2,3,4,5,6,7::3,5,1,0,4,2,7,6::4,0,2,3,5,7,6,1

# 8b4b64e6edb788efc8d29b514b699ef0
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1042 --custom_alignment=1,0,2,3,7,5,6,4::7,1,0,3,4,5,6,2::1,2,3,5,0,4,6,7::3,6,1,2,5,4,0,7::6,7,3,2,0,4,5,1::1,0,2,3,4,5,6,7::3,5,1,0,4,2,7,6::4,0,2,3,5,7,6,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8b4b64e6edb788efc8d29b514b699ef0
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=337 --custom_alignment=1,0,2,3,7,5,6,4::7,1,0,3,4,5,6,2::1,2,3,5,0,4,6,7::3,6,1,2,5,4,0,7::6,7,3,2,0,4,5,1::1,0,2,3,4,5,6,7::3,5,1,0,4,2,7,6::4,0,2,3,5,7,6,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8b4b64e6edb788efc8d29b514b699ef0
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=4138 --custom_alignment=1,0,2,3,7,5,6,4::7,1,0,3,4,5,6,2::1,2,3,5,0,4,6,7::3,6,1,2,5,4,0,7::6,7,3,2,0,4,5,1::1,0,2,3,4,5,6,7::3,5,1,0,4,2,7,6::4,0,2,3,5,7,6,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8b4b64e6edb788efc8d29b514b699ef0
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=7761 --custom_alignment=1,0,2,3,7,5,6,4::7,1,0,3,4,5,6,2::1,2,3,5,0,4,6,7::3,6,1,2,5,4,0,7::6,7,3,2,0,4,5,1::1,0,2,3,4,5,6,7::3,5,1,0,4,2,7,6::4,0,2,3,5,7,6,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8b4b64e6edb788efc8d29b514b699ef0
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1859 --custom_alignment=1,0,2,3,7,5,6,4::7,1,0,3,4,5,6,2::1,2,3,5,0,4,6,7::3,6,1,2,5,4,0,7::6,7,3,2,0,4,5,1::1,0,2,3,4,5,6,7::3,5,1,0,4,2,7,6::4,0,2,3,5,7,6,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 7,1,2,3,4,5,6,0::1,2,0,4,7,5,6,3::3,6,1,5,4,0,2,7::1,6,3,5,4,7,2,0::6,2,0,4,5,1,3,7::0,7,2,5,3,4,1,6::2,0,7,3,4,5,1,6::2,3,0,4,6,5,1,7

# 010fe7ded85ea075dd045f067458bd4b
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8411 --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,4,7,5,6,3::3,6,1,5,4,0,2,7::1,6,3,5,4,7,2,0::6,2,0,4,5,1,3,7::0,7,2,5,3,4,1,6::2,0,7,3,4,5,1,6::2,3,0,4,6,5,1,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 010fe7ded85ea075dd045f067458bd4b
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2453 --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,4,7,5,6,3::3,6,1,5,4,0,2,7::1,6,3,5,4,7,2,0::6,2,0,4,5,1,3,7::0,7,2,5,3,4,1,6::2,0,7,3,4,5,1,6::2,3,0,4,6,5,1,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 010fe7ded85ea075dd045f067458bd4b
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9829 --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,4,7,5,6,3::3,6,1,5,4,0,2,7::1,6,3,5,4,7,2,0::6,2,0,4,5,1,3,7::0,7,2,5,3,4,1,6::2,0,7,3,4,5,1,6::2,3,0,4,6,5,1,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 010fe7ded85ea075dd045f067458bd4b
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2649 --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,4,7,5,6,3::3,6,1,5,4,0,2,7::1,6,3,5,4,7,2,0::6,2,0,4,5,1,3,7::0,7,2,5,3,4,1,6::2,0,7,3,4,5,1,6::2,3,0,4,6,5,1,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 010fe7ded85ea075dd045f067458bd4b
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1206 --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,4,7,5,6,3::3,6,1,5,4,0,2,7::1,6,3,5,4,7,2,0::6,2,0,4,5,1,3,7::0,7,2,5,3,4,1,6::2,0,7,3,4,5,1,6::2,3,0,4,6,5,1,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 2,1,4,0,3,5,6,7::6,2,7,3,4,5,0,1::1,3,6,0,4,5,7,2::1,2,5,3,4,0,7,6::4,5,2,1,7,6,0,3::1,0,2,4,5,3,6,7::4,6,1,5,2,7,0,3::2,3,0,1,4,5,6,7

# 8841e81d5900c3cd750f7625fc6e83b0
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2593 --custom_alignment=2,1,4,0,3,5,6,7::6,2,7,3,4,5,0,1::1,3,6,0,4,5,7,2::1,2,5,3,4,0,7,6::4,5,2,1,7,6,0,3::1,0,2,4,5,3,6,7::4,6,1,5,2,7,0,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8841e81d5900c3cd750f7625fc6e83b0
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1326 --custom_alignment=2,1,4,0,3,5,6,7::6,2,7,3,4,5,0,1::1,3,6,0,4,5,7,2::1,2,5,3,4,0,7,6::4,5,2,1,7,6,0,3::1,0,2,4,5,3,6,7::4,6,1,5,2,7,0,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8841e81d5900c3cd750f7625fc6e83b0
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9361 --custom_alignment=2,1,4,0,3,5,6,7::6,2,7,3,4,5,0,1::1,3,6,0,4,5,7,2::1,2,5,3,4,0,7,6::4,5,2,1,7,6,0,3::1,0,2,4,5,3,6,7::4,6,1,5,2,7,0,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8841e81d5900c3cd750f7625fc6e83b0
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1613 --custom_alignment=2,1,4,0,3,5,6,7::6,2,7,3,4,5,0,1::1,3,6,0,4,5,7,2::1,2,5,3,4,0,7,6::4,5,2,1,7,6,0,3::1,0,2,4,5,3,6,7::4,6,1,5,2,7,0,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 8841e81d5900c3cd750f7625fc6e83b0
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5596 --custom_alignment=2,1,4,0,3,5,6,7::6,2,7,3,4,5,0,1::1,3,6,0,4,5,7,2::1,2,5,3,4,0,7,6::4,5,2,1,7,6,0,3::1,0,2,4,5,3,6,7::4,6,1,5,2,7,0,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 0,7,6,5,3,2,4,1::5,2,3,4,1,0,6,7::3,1,2,0,4,6,5,7::7,4,1,2,0,5,6,3::7,2,3,5,0,6,4,1::1,4,2,6,3,5,0,7::4,0,1,3,2,5,6,7::2,4,0,1,3,5,6,7

# 0ef1b0953b9e880b3f88f5ba0e7efe15
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=363 --custom_alignment=0,7,6,5,3,2,4,1::5,2,3,4,1,0,6,7::3,1,2,0,4,6,5,7::7,4,1,2,0,5,6,3::7,2,3,5,0,6,4,1::1,4,2,6,3,5,0,7::4,0,1,3,2,5,6,7::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0ef1b0953b9e880b3f88f5ba0e7efe15
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5949 --custom_alignment=0,7,6,5,3,2,4,1::5,2,3,4,1,0,6,7::3,1,2,0,4,6,5,7::7,4,1,2,0,5,6,3::7,2,3,5,0,6,4,1::1,4,2,6,3,5,0,7::4,0,1,3,2,5,6,7::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0ef1b0953b9e880b3f88f5ba0e7efe15
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8443 --custom_alignment=0,7,6,5,3,2,4,1::5,2,3,4,1,0,6,7::3,1,2,0,4,6,5,7::7,4,1,2,0,5,6,3::7,2,3,5,0,6,4,1::1,4,2,6,3,5,0,7::4,0,1,3,2,5,6,7::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0ef1b0953b9e880b3f88f5ba0e7efe15
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=806 --custom_alignment=0,7,6,5,3,2,4,1::5,2,3,4,1,0,6,7::3,1,2,0,4,6,5,7::7,4,1,2,0,5,6,3::7,2,3,5,0,6,4,1::1,4,2,6,3,5,0,7::4,0,1,3,2,5,6,7::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0ef1b0953b9e880b3f88f5ba0e7efe15
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2456 --custom_alignment=0,7,6,5,3,2,4,1::5,2,3,4,1,0,6,7::3,1,2,0,4,6,5,7::7,4,1,2,0,5,6,3::7,2,3,5,0,6,4,1::1,4,2,6,3,5,0,7::4,0,1,3,2,5,6,7::2,4,0,1,3,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 4,1,3,0,2,7,6,5::1,0,6,3,4,7,2,5::1,0,3,4,2,6,5,7::3,2,1,4,5,0,7,6::3,2,6,4,5,7,0,1::2,0,1,3,4,6,7,5::2,0,1,3,4,5,6,7::7,4,0,3,6,5,1,2

# f45dc5c1b0fa7b8a857e384c23b54769
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2722 --custom_alignment=4,1,3,0,2,7,6,5::1,0,6,3,4,7,2,5::1,0,3,4,2,6,5,7::3,2,1,4,5,0,7,6::3,2,6,4,5,7,0,1::2,0,1,3,4,6,7,5::2,0,1,3,4,5,6,7::7,4,0,3,6,5,1,2 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f45dc5c1b0fa7b8a857e384c23b54769
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9671 --custom_alignment=4,1,3,0,2,7,6,5::1,0,6,3,4,7,2,5::1,0,3,4,2,6,5,7::3,2,1,4,5,0,7,6::3,2,6,4,5,7,0,1::2,0,1,3,4,6,7,5::2,0,1,3,4,5,6,7::7,4,0,3,6,5,1,2 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f45dc5c1b0fa7b8a857e384c23b54769
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=4879 --custom_alignment=4,1,3,0,2,7,6,5::1,0,6,3,4,7,2,5::1,0,3,4,2,6,5,7::3,2,1,4,5,0,7,6::3,2,6,4,5,7,0,1::2,0,1,3,4,6,7,5::2,0,1,3,4,5,6,7::7,4,0,3,6,5,1,2 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f45dc5c1b0fa7b8a857e384c23b54769
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5838 --custom_alignment=4,1,3,0,2,7,6,5::1,0,6,3,4,7,2,5::1,0,3,4,2,6,5,7::3,2,1,4,5,0,7,6::3,2,6,4,5,7,0,1::2,0,1,3,4,6,7,5::2,0,1,3,4,5,6,7::7,4,0,3,6,5,1,2 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# f45dc5c1b0fa7b8a857e384c23b54769
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5447 --custom_alignment=4,1,3,0,2,7,6,5::1,0,6,3,4,7,2,5::1,0,3,4,2,6,5,7::3,2,1,4,5,0,7,6::3,2,6,4,5,7,0,1::2,0,1,3,4,6,7,5::2,0,1,3,4,5,6,7::7,4,0,3,6,5,1,2 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 4,1,2,5,0,3,6,7::7,6,2,5,0,3,4,1::1,2,4,3,0,5,6,7::7,0,2,3,1,4,6,5::0,3,2,4,5,6,1,7::4,2,1,7,0,6,5,3::2,3,6,4,5,0,1,7::2,3,0,1,4,5,6,7

# 9f4c20615992122a2ae5b2bbf550c25e
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=6312 --custom_alignment=4,1,2,5,0,3,6,7::7,6,2,5,0,3,4,1::1,2,4,3,0,5,6,7::7,0,2,3,1,4,6,5::0,3,2,4,5,6,1,7::4,2,1,7,0,6,5,3::2,3,6,4,5,0,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9f4c20615992122a2ae5b2bbf550c25e
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2246 --custom_alignment=4,1,2,5,0,3,6,7::7,6,2,5,0,3,4,1::1,2,4,3,0,5,6,7::7,0,2,3,1,4,6,5::0,3,2,4,5,6,1,7::4,2,1,7,0,6,5,3::2,3,6,4,5,0,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9f4c20615992122a2ae5b2bbf550c25e
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9505 --custom_alignment=4,1,2,5,0,3,6,7::7,6,2,5,0,3,4,1::1,2,4,3,0,5,6,7::7,0,2,3,1,4,6,5::0,3,2,4,5,6,1,7::4,2,1,7,0,6,5,3::2,3,6,4,5,0,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9f4c20615992122a2ae5b2bbf550c25e
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=3683 --custom_alignment=4,1,2,5,0,3,6,7::7,6,2,5,0,3,4,1::1,2,4,3,0,5,6,7::7,0,2,3,1,4,6,5::0,3,2,4,5,6,1,7::4,2,1,7,0,6,5,3::2,3,6,4,5,0,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 9f4c20615992122a2ae5b2bbf550c25e
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2081 --custom_alignment=4,1,2,5,0,3,6,7::7,6,2,5,0,3,4,1::1,2,4,3,0,5,6,7::7,0,2,3,1,4,6,5::0,3,2,4,5,6,1,7::4,2,1,7,0,6,5,3::2,3,6,4,5,0,1,7::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 4,1,0,3,6,5,2,7::5,2,7,3,4,1,6,0::2,7,1,4,5,6,0,3::3,4,2,5,0,7,6,1::1,2,7,3,5,0,4,6::5,2,0,7,4,1,6,3::2,0,1,4,5,7,6,3::2,3,0,1,4,5,6,7

# c103424b973f347426d7e3da1904f2c2
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1803 --custom_alignment=4,1,0,3,6,5,2,7::5,2,7,3,4,1,6,0::2,7,1,4,5,6,0,3::3,4,2,5,0,7,6,1::1,2,7,3,5,0,4,6::5,2,0,7,4,1,6,3::2,0,1,4,5,7,6,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c103424b973f347426d7e3da1904f2c2
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=7550 --custom_alignment=4,1,0,3,6,5,2,7::5,2,7,3,4,1,6,0::2,7,1,4,5,6,0,3::3,4,2,5,0,7,6,1::1,2,7,3,5,0,4,6::5,2,0,7,4,1,6,3::2,0,1,4,5,7,6,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c103424b973f347426d7e3da1904f2c2
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2419 --custom_alignment=4,1,0,3,6,5,2,7::5,2,7,3,4,1,6,0::2,7,1,4,5,6,0,3::3,4,2,5,0,7,6,1::1,2,7,3,5,0,4,6::5,2,0,7,4,1,6,3::2,0,1,4,5,7,6,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c103424b973f347426d7e3da1904f2c2
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9078 --custom_alignment=4,1,0,3,6,5,2,7::5,2,7,3,4,1,6,0::2,7,1,4,5,6,0,3::3,4,2,5,0,7,6,1::1,2,7,3,5,0,4,6::5,2,0,7,4,1,6,3::2,0,1,4,5,7,6,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# c103424b973f347426d7e3da1904f2c2
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5755 --custom_alignment=4,1,0,3,6,5,2,7::5,2,7,3,4,1,6,0::2,7,1,4,5,6,0,3::3,4,2,5,0,7,6,1::1,2,7,3,5,0,4,6::5,2,0,7,4,1,6,3::2,0,1,4,5,7,6,3::2,3,0,1,4,5,6,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 0,7,2,6,5,3,1,4::1,3,0,2,4,5,7,6::1,7,6,5,4,0,3,2::2,0,3,4,5,1,6,7::1,7,3,4,5,6,0,2::0,3,7,2,6,5,4,1::6,3,1,0,2,5,4,7::7,2,0,3,4,5,6,1

# 0c8f60abc8261940be0b329b4603c016
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9376 --custom_alignment=0,7,2,6,5,3,1,4::1,3,0,2,4,5,7,6::1,7,6,5,4,0,3,2::2,0,3,4,5,1,6,7::1,7,3,4,5,6,0,2::0,3,7,2,6,5,4,1::6,3,1,0,2,5,4,7::7,2,0,3,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0c8f60abc8261940be0b329b4603c016
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=4246 --custom_alignment=0,7,2,6,5,3,1,4::1,3,0,2,4,5,7,6::1,7,6,5,4,0,3,2::2,0,3,4,5,1,6,7::1,7,3,4,5,6,0,2::0,3,7,2,6,5,4,1::6,3,1,0,2,5,4,7::7,2,0,3,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0c8f60abc8261940be0b329b4603c016
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5095 --custom_alignment=0,7,2,6,5,3,1,4::1,3,0,2,4,5,7,6::1,7,6,5,4,0,3,2::2,0,3,4,5,1,6,7::1,7,3,4,5,6,0,2::0,3,7,2,6,5,4,1::6,3,1,0,2,5,4,7::7,2,0,3,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0c8f60abc8261940be0b329b4603c016
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=4523 --custom_alignment=0,7,2,6,5,3,1,4::1,3,0,2,4,5,7,6::1,7,6,5,4,0,3,2::2,0,3,4,5,1,6,7::1,7,3,4,5,6,0,2::0,3,7,2,6,5,4,1::6,3,1,0,2,5,4,7::7,2,0,3,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 0c8f60abc8261940be0b329b4603c016
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8215 --custom_alignment=0,7,2,6,5,3,1,4::1,3,0,2,4,5,7,6::1,7,6,5,4,0,3,2::2,0,3,4,5,1,6,7::1,7,3,4,5,6,0,2::0,3,7,2,6,5,4,1::6,3,1,0,2,5,4,7::7,2,0,3,4,5,6,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 0,1,3,2,5,4,7,6::4,2,7,3,1,5,6,0::2,1,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,3,2,0,4,6,5,7::1,7,3,4,0,6,2,5::7,0,1,5,4,3,6,2::0,6,7,4,2,5,3,1

# 6868dbaa98511ccacd6717036ee7075b
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2791 --custom_alignment=0,1,3,2,5,4,7,6::4,2,7,3,1,5,6,0::2,1,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,3,2,0,4,6,5,7::1,7,3,4,0,6,2,5::7,0,1,5,4,3,6,2::0,6,7,4,2,5,3,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6868dbaa98511ccacd6717036ee7075b
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=2271 --custom_alignment=0,1,3,2,5,4,7,6::4,2,7,3,1,5,6,0::2,1,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,3,2,0,4,6,5,7::1,7,3,4,0,6,2,5::7,0,1,5,4,3,6,2::0,6,7,4,2,5,3,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6868dbaa98511ccacd6717036ee7075b
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=7149 --custom_alignment=0,1,3,2,5,4,7,6::4,2,7,3,1,5,6,0::2,1,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,3,2,0,4,6,5,7::1,7,3,4,0,6,2,5::7,0,1,5,4,3,6,2::0,6,7,4,2,5,3,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6868dbaa98511ccacd6717036ee7075b
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5739 --custom_alignment=0,1,3,2,5,4,7,6::4,2,7,3,1,5,6,0::2,1,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,3,2,0,4,6,5,7::1,7,3,4,0,6,2,5::7,0,1,5,4,3,6,2::0,6,7,4,2,5,3,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6868dbaa98511ccacd6717036ee7075b
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=3261 --custom_alignment=0,1,3,2,5,4,7,6::4,2,7,3,1,5,6,0::2,1,3,0,4,5,6,7::0,2,3,4,5,1,6,7::1,3,2,0,4,6,5,7::1,7,3,4,0,6,2,5::7,0,1,5,4,3,6,2::0,6,7,4,2,5,3,1 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 0,1,2,6,4,5,7,3::3,4,1,0,2,7,6,5::1,5,4,6,2,3,0,7::5,2,3,4,1,0,6,7::7,5,3,0,2,6,1,4::1,4,2,3,7,6,5,0::6,2,1,4,0,5,3,7::2,3,0,7,4,5,1,6

# a8cd0a9e37a42ad6443292260c96ba7a
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1520 --custom_alignment=0,1,2,6,4,5,7,3::3,4,1,0,2,7,6,5::1,5,4,6,2,3,0,7::5,2,3,4,1,0,6,7::7,5,3,0,2,6,1,4::1,4,2,3,7,6,5,0::6,2,1,4,0,5,3,7::2,3,0,7,4,5,1,6 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a8cd0a9e37a42ad6443292260c96ba7a
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5329 --custom_alignment=0,1,2,6,4,5,7,3::3,4,1,0,2,7,6,5::1,5,4,6,2,3,0,7::5,2,3,4,1,0,6,7::7,5,3,0,2,6,1,4::1,4,2,3,7,6,5,0::6,2,1,4,0,5,3,7::2,3,0,7,4,5,1,6 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a8cd0a9e37a42ad6443292260c96ba7a
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=3741 --custom_alignment=0,1,2,6,4,5,7,3::3,4,1,0,2,7,6,5::1,5,4,6,2,3,0,7::5,2,3,4,1,0,6,7::7,5,3,0,2,6,1,4::1,4,2,3,7,6,5,0::6,2,1,4,0,5,3,7::2,3,0,7,4,5,1,6 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a8cd0a9e37a42ad6443292260c96ba7a
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=6415 --custom_alignment=0,1,2,6,4,5,7,3::3,4,1,0,2,7,6,5::1,5,4,6,2,3,0,7::5,2,3,4,1,0,6,7::7,5,3,0,2,6,1,4::1,4,2,3,7,6,5,0::6,2,1,4,0,5,3,7::2,3,0,7,4,5,1,6 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# a8cd0a9e37a42ad6443292260c96ba7a
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=6434 --custom_alignment=0,1,2,6,4,5,7,3::3,4,1,0,2,7,6,5::1,5,4,6,2,3,0,7::5,2,3,4,1,0,6,7::7,5,3,0,2,6,1,4::1,4,2,3,7,6,5,0::6,2,1,4,0,5,3,7::2,3,0,7,4,5,1,6 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 5,0,2,3,4,1,6,7::4,5,3,2,0,6,7,1::4,1,3,0,2,5,6,7::1,2,3,4,0,5,6,7::1,5,4,7,2,6,0,3::3,0,2,1,4,5,6,7::0,2,3,5,4,6,1,7::1,0,6,7,4,2,5,3

# 580d2cd4318006185f07f0023942844f
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5572 --custom_alignment=5,0,2,3,4,1,6,7::4,5,3,2,0,6,7,1::4,1,3,0,2,5,6,7::1,2,3,4,0,5,6,7::1,5,4,7,2,6,0,3::3,0,2,1,4,5,6,7::0,2,3,5,4,6,1,7::1,0,6,7,4,2,5,3 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 580d2cd4318006185f07f0023942844f
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8499 --custom_alignment=5,0,2,3,4,1,6,7::4,5,3,2,0,6,7,1::4,1,3,0,2,5,6,7::1,2,3,4,0,5,6,7::1,5,4,7,2,6,0,3::3,0,2,1,4,5,6,7::0,2,3,5,4,6,1,7::1,0,6,7,4,2,5,3 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 580d2cd4318006185f07f0023942844f
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1643 --custom_alignment=5,0,2,3,4,1,6,7::4,5,3,2,0,6,7,1::4,1,3,0,2,5,6,7::1,2,3,4,0,5,6,7::1,5,4,7,2,6,0,3::3,0,2,1,4,5,6,7::0,2,3,5,4,6,1,7::1,0,6,7,4,2,5,3 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 580d2cd4318006185f07f0023942844f
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=5371 --custom_alignment=5,0,2,3,4,1,6,7::4,5,3,2,0,6,7,1::4,1,3,0,2,5,6,7::1,2,3,4,0,5,6,7::1,5,4,7,2,6,0,3::3,0,2,1,4,5,6,7::0,2,3,5,4,6,1,7::1,0,6,7,4,2,5,3 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 580d2cd4318006185f07f0023942844f
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=6693 --custom_alignment=5,0,2,3,4,1,6,7::4,5,3,2,0,6,7,1::4,1,3,0,2,5,6,7::1,2,3,4,0,5,6,7::1,5,4,7,2,6,0,3::3,0,2,1,4,5,6,7::0,2,3,5,4,6,1,7::1,0,6,7,4,2,5,3 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 2,1,0,4,3,7,6,5::6,2,7,3,4,1,5,0::1,6,7,3,4,2,5,0::0,2,3,4,1,5,6,7::1,3,2,0,5,6,4,7::5,3,2,6,4,0,1,7::2,5,0,7,4,1,6,3::2,3,6,4,1,5,0,7

# 6f9088ff5e358479356f2cce7414b0d5
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=6680 --custom_alignment=2,1,0,4,3,7,6,5::6,2,7,3,4,1,5,0::1,6,7,3,4,2,5,0::0,2,3,4,1,5,6,7::1,3,2,0,5,6,4,7::5,3,2,6,4,0,1,7::2,5,0,7,4,1,6,3::2,3,6,4,1,5,0,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6f9088ff5e358479356f2cce7414b0d5
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=995 --custom_alignment=2,1,0,4,3,7,6,5::6,2,7,3,4,1,5,0::1,6,7,3,4,2,5,0::0,2,3,4,1,5,6,7::1,3,2,0,5,6,4,7::5,3,2,6,4,0,1,7::2,5,0,7,4,1,6,3::2,3,6,4,1,5,0,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6f9088ff5e358479356f2cce7414b0d5
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8009 --custom_alignment=2,1,0,4,3,7,6,5::6,2,7,3,4,1,5,0::1,6,7,3,4,2,5,0::0,2,3,4,1,5,6,7::1,3,2,0,5,6,4,7::5,3,2,6,4,0,1,7::2,5,0,7,4,1,6,3::2,3,6,4,1,5,0,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6f9088ff5e358479356f2cce7414b0d5
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=8448 --custom_alignment=2,1,0,4,3,7,6,5::6,2,7,3,4,1,5,0::1,6,7,3,4,2,5,0::0,2,3,4,1,5,6,7::1,3,2,0,5,6,4,7::5,3,2,6,4,0,1,7::2,5,0,7,4,1,6,3::2,3,6,4,1,5,0,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# 6f9088ff5e358479356f2cce7414b0d5
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=332 --custom_alignment=2,1,0,4,3,7,6,5::6,2,7,3,4,1,5,0::1,6,7,3,4,2,5,0::0,2,3,4,1,5,6,7::1,3,2,0,5,6,4,7::5,3,2,6,4,0,1,7::2,5,0,7,4,1,6,3::2,3,6,4,1,5,0,7 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# mutate 32 alignment for 9xx: 4,3,2,1,7,5,6,0::1,2,4,5,0,3,6,7::1,2,3,7,6,5,4,0::1,2,0,4,5,3,6,7::2,1,4,3,0,6,5,7::2,1,3,5,0,4,6,7::2,1,0,4,5,3,6,7::0,1,4,3,6,7,2,5

# bfccdc9c43a7e1d9d0324382f5460064
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=3248 --custom_alignment=4,3,2,1,7,5,6,0::1,2,4,5,0,3,6,7::1,2,3,7,6,5,4,0::1,2,0,4,5,3,6,7::2,1,4,3,0,6,5,7::2,1,3,5,0,4,6,7::2,1,0,4,5,3,6,7::0,1,4,3,6,7,2,5 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# bfccdc9c43a7e1d9d0324382f5460064
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=3636 --custom_alignment=4,3,2,1,7,5,6,0::1,2,4,5,0,3,6,7::1,2,3,7,6,5,4,0::1,2,0,4,5,3,6,7::2,1,4,3,0,6,5,7::2,1,3,5,0,4,6,7::2,1,0,4,5,3,6,7::0,1,4,3,6,7,2,5 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# bfccdc9c43a7e1d9d0324382f5460064
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9429 --custom_alignment=4,3,2,1,7,5,6,0::1,2,4,5,0,3,6,7::1,2,3,7,6,5,4,0::1,2,0,4,5,3,6,7::2,1,4,3,0,6,5,7::2,1,3,5,0,4,6,7::2,1,0,4,5,3,6,7::0,1,4,3,6,7,2,5 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# bfccdc9c43a7e1d9d0324382f5460064
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=1416 --custom_alignment=4,3,2,1,7,5,6,0::1,2,4,5,0,3,6,7::1,2,3,7,6,5,4,0::1,2,0,4,5,3,6,7::2,1,4,3,0,6,5,7::2,1,3,5,0,4,6,7::2,1,0,4,5,3,6,7::0,1,4,3,6,7,2,5 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907

# bfccdc9c43a7e1d9d0324382f5460064
sbatch -J search_9xx_mutate_32 submit.sh python 1.train.py --seed=9506 --custom_alignment=4,3,2,1,7,5,6,0::1,2,4,5,0,3,6,7::1,2,3,7,6,5,4,0::1,2,0,4,5,3,6,7::2,1,4,3,0,6,5,7::2,1,3,5,0,4,6,7::2,1,0,4,5,3,6,7::0,1,4,3,6,7,2,5 --tensorboard=tensorboard/9xx_mutate_32 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907
# Total jobs: 500.
