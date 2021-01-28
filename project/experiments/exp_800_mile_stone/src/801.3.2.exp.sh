#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="801.3.train"

description="
For each type,
we randomly pick two robots, and train a RL policy on them, with the observation and action space aligned, we measure the learnability at 2e6 steps.
for the same two robots, we train another RL policy on them, with the observation and action space randomized, we measure the learnability at 2e6 steps.
We repeat the two experiments 10 times, so we have a comparison between aligned and randomized when training on 2 robots.
We then repeat the same procedures on 4 robots, 8 robots, 16 robots, and have comparisons for those conditions.
"



# train on bodies:  [311 304]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2732 -f=$exp_name/2_aligned/ --train_bodies=311,304 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [315 307]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9845 -f=$exp_name/2_aligned/ --train_bodies=315,307 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [313 309]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3264 -f=$exp_name/2_aligned/ --train_bodies=313,309 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [303 314]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4859 -f=$exp_name/2_aligned/ --train_bodies=303,314 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [300 309]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9225 -f=$exp_name/2_aligned/ --train_bodies=300,309 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [311 315]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7891 -f=$exp_name/2_aligned/ --train_bodies=311,315 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [314 307]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4373 -f=$exp_name/2_aligned/ --train_bodies=314,307 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [302 303]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5874 -f=$exp_name/2_aligned/ --train_bodies=302,303 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [312 300]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6744 -f=$exp_name/2_aligned/ --train_bodies=312,300 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [311 309]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3468 -f=$exp_name/2_aligned/ --train_bodies=311,309 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [400 413]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=235 -f=$exp_name/2_aligned/ --train_bodies=400,413 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [404 402]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5192 -f=$exp_name/2_aligned/ --train_bodies=404,402 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [401 400]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=905 -f=$exp_name/2_aligned/ --train_bodies=401,400 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [402 408]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7813 -f=$exp_name/2_aligned/ --train_bodies=402,408 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [404 403]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2895 -f=$exp_name/2_aligned/ --train_bodies=404,403 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [402 413]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5056 -f=$exp_name/2_aligned/ --train_bodies=402,413 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [415 413]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=144 -f=$exp_name/2_aligned/ --train_bodies=415,413 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [404 400]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4225 -f=$exp_name/2_aligned/ --train_bodies=404,400 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [401 402]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7751 -f=$exp_name/2_aligned/ --train_bodies=401,402 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [414 400]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3462 -f=$exp_name/2_aligned/ --train_bodies=414,400 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [503 506]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7336 -f=$exp_name/2_aligned/ --train_bodies=503,506 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [510 507]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2575 -f=$exp_name/2_aligned/ --train_bodies=510,507 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [515 504]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6637 -f=$exp_name/2_aligned/ --train_bodies=515,504 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [507 500]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2514 -f=$exp_name/2_aligned/ --train_bodies=507,500 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [515 503]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1099 -f=$exp_name/2_aligned/ --train_bodies=515,503 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [502 504]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4770 -f=$exp_name/2_aligned/ --train_bodies=502,504 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [510 505]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=433 -f=$exp_name/2_aligned/ --train_bodies=510,505 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [508 509]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6751 -f=$exp_name/2_aligned/ --train_bodies=508,509 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [511 513]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2773 -f=$exp_name/2_aligned/ --train_bodies=511,513 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [506 508]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5167 -f=$exp_name/2_aligned/ --train_bodies=506,508 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [614 613]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5994 -f=$exp_name/2_aligned/ --train_bodies=614,613 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [613 612]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1688 -f=$exp_name/2_aligned/ --train_bodies=613,612 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [615 614]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9859 -f=$exp_name/2_aligned/ --train_bodies=615,614 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [612 615]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9160 -f=$exp_name/2_aligned/ --train_bodies=612,615 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [615 603]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6400 -f=$exp_name/2_aligned/ --train_bodies=615,603 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [607 602]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8981 -f=$exp_name/2_aligned/ --train_bodies=607,602 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [604 606]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2707 -f=$exp_name/2_aligned/ --train_bodies=604,606 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [605 610]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7161 -f=$exp_name/2_aligned/ --train_bodies=605,610 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [613 610]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1705 -f=$exp_name/2_aligned/ --train_bodies=613,610 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [604 601]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7818 -f=$exp_name/2_aligned/ --train_bodies=604,601 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [300 314]

# ==> Alignment :  3,5,2,0,4,1::3,4,0,2,5,1
# 2d02d8ee30b8d67d899c5895d7219540
sbatch -J $exp_name $submit_to python 1.train.py --seed=1146 -f=$exp_name/2_randomized/ --train_bodies=300,314 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,5,2,0,4,1::3,4,0,2,5,1 --custom_align_max_joints=6

# train on bodies:  [301 308]

# ==> Alignment :  1,0,5,2,3,4::0,5,4,3,2,1
# bb605d4d97a48f801e73d6de8c0bba31
sbatch -J $exp_name $submit_to python 1.train.py --seed=8366 -f=$exp_name/2_randomized/ --train_bodies=301,308 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,0,5,2,3,4::0,5,4,3,2,1 --custom_align_max_joints=6

# train on bodies:  [308 315]

# ==> Alignment :  2,1,4,0,3,5::1,4,0,2,5,3
# e06910d9b24df73b498be876211c13cc
sbatch -J $exp_name $submit_to python 1.train.py --seed=709 -f=$exp_name/2_randomized/ --train_bodies=308,315 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,1,4,0,3,5::1,4,0,2,5,3 --custom_align_max_joints=6

# train on bodies:  [304 300]

# ==> Alignment :  0,4,1,3,5,2::0,4,2,5,3,1
# b6b5caa4763d33e8d90241493de9bf53
sbatch -J $exp_name $submit_to python 1.train.py --seed=6017 -f=$exp_name/2_randomized/ --train_bodies=304,300 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,4,1,3,5,2::0,4,2,5,3,1 --custom_align_max_joints=6

# train on bodies:  [307 302]

# ==> Alignment :  0,4,2,1,5,3::3,1,4,0,2,5
# 4382efbc6da4ad204f7b1331e4f09acf
sbatch -J $exp_name $submit_to python 1.train.py --seed=456 -f=$exp_name/2_randomized/ --train_bodies=307,302 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,4,2,1,5,3::3,1,4,0,2,5 --custom_align_max_joints=6

# train on bodies:  [310 301]

# ==> Alignment :  1,5,0,3,4,2::5,4,2,0,1,3
# 0849c3ac5cbf60768dad44e3f5287463
sbatch -J $exp_name $submit_to python 1.train.py --seed=6962 -f=$exp_name/2_randomized/ --train_bodies=310,301 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,5,0,3,4,2::5,4,2,0,1,3 --custom_align_max_joints=6

# train on bodies:  [305 308]

# ==> Alignment :  0,3,2,4,5,1::0,1,3,4,5,2
# 92fc2a0c30953ce5d16a12f947a0b7d3
sbatch -J $exp_name $submit_to python 1.train.py --seed=9274 -f=$exp_name/2_randomized/ --train_bodies=305,308 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,3,2,4,5,1::0,1,3,4,5,2 --custom_align_max_joints=6

# train on bodies:  [311 302]

# ==> Alignment :  5,0,1,2,3,4::1,2,0,5,3,4
# a2a93e838948319142051f3943ece946
sbatch -J $exp_name $submit_to python 1.train.py --seed=8039 -f=$exp_name/2_randomized/ --train_bodies=311,302 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,0,1,2,3,4::1,2,0,5,3,4 --custom_align_max_joints=6

# train on bodies:  [305 303]

# ==> Alignment :  5,2,3,4,1,0::4,1,0,2,3,5
# 41d30f59dce937f7ce40338704e5922c
sbatch -J $exp_name $submit_to python 1.train.py --seed=3678 -f=$exp_name/2_randomized/ --train_bodies=305,303 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,2,3,4,1,0::4,1,0,2,3,5 --custom_align_max_joints=6

# train on bodies:  [303 302]

# ==> Alignment :  1,5,4,3,0,2::0,1,3,5,2,4
# b66af14be49a530ff966f5c4aeaa2b45
sbatch -J $exp_name $submit_to python 1.train.py --seed=7844 -f=$exp_name/2_randomized/ --train_bodies=303,302 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,5,4,3,0,2::0,1,3,5,2,4 --custom_align_max_joints=6

# train on bodies:  [415 404]

# ==> Alignment :  3,5,0,4,1,2::1,0,4,5,2,3
# fddb991900b10113d8bdad3f11c937ab
sbatch -J $exp_name $submit_to python 1.train.py --seed=2915 -f=$exp_name/2_randomized/ --train_bodies=415,404 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,5,0,4,1,2::1,0,4,5,2,3 --custom_align_max_joints=6

# train on bodies:  [404 408]

# ==> Alignment :  0,5,4,2,1,3::3,0,4,1,2,5
# 62cad062aa72681ea91758d6bd266a2b
sbatch -J $exp_name $submit_to python 1.train.py --seed=2254 -f=$exp_name/2_randomized/ --train_bodies=404,408 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,5,4,2,1,3::3,0,4,1,2,5 --custom_align_max_joints=6

# train on bodies:  [412 402]

# ==> Alignment :  4,2,5,0,3,1::5,1,0,3,2,4
# f6da917efb570e36516018532b959d1b
sbatch -J $exp_name $submit_to python 1.train.py --seed=4079 -f=$exp_name/2_randomized/ --train_bodies=412,402 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,2,5,0,3,1::5,1,0,3,2,4 --custom_align_max_joints=6

# train on bodies:  [406 412]

# ==> Alignment :  3,2,0,4,1,5::0,1,2,4,5,3
# 835f56c55ae4fda0ec21912085287c7d
sbatch -J $exp_name $submit_to python 1.train.py --seed=9917 -f=$exp_name/2_randomized/ --train_bodies=406,412 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,2,0,4,1,5::0,1,2,4,5,3 --custom_align_max_joints=6

# train on bodies:  [410 414]

# ==> Alignment :  2,4,5,0,1,3::4,2,0,5,1,3
# ef941979c52ba1db4254fea1558c976b
sbatch -J $exp_name $submit_to python 1.train.py --seed=3046 -f=$exp_name/2_randomized/ --train_bodies=410,414 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,5,0,1,3::4,2,0,5,1,3 --custom_align_max_joints=6

# train on bodies:  [409 401]

# ==> Alignment :  0,1,4,2,3,5::3,5,1,0,2,4
# 9d5c0e92d7122c9f0256e09374e2dea8
sbatch -J $exp_name $submit_to python 1.train.py --seed=7286 -f=$exp_name/2_randomized/ --train_bodies=409,401 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,1,4,2,3,5::3,5,1,0,2,4 --custom_align_max_joints=6

# train on bodies:  [402 412]

# ==> Alignment :  0,2,5,1,4,3::4,5,3,1,0,2
# 1c7193b0deff0cc51cfaa35083aa4a97
sbatch -J $exp_name $submit_to python 1.train.py --seed=5520 -f=$exp_name/2_randomized/ --train_bodies=402,412 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,5,1,4,3::4,5,3,1,0,2 --custom_align_max_joints=6

# train on bodies:  [405 406]

# ==> Alignment :  4,5,1,3,2,0::2,0,3,4,5,1
# 5aab765eb81348e12fa4e0815bb5f715
sbatch -J $exp_name $submit_to python 1.train.py --seed=1032 -f=$exp_name/2_randomized/ --train_bodies=405,406 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,5,1,3,2,0::2,0,3,4,5,1 --custom_align_max_joints=6

# train on bodies:  [409 411]

# ==> Alignment :  3,0,4,1,2,5::4,2,5,3,0,1
# b7c14f1b7a06443eacf22eab6570159d
sbatch -J $exp_name $submit_to python 1.train.py --seed=740 -f=$exp_name/2_randomized/ --train_bodies=409,411 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,0,4,1,2,5::4,2,5,3,0,1 --custom_align_max_joints=6

# train on bodies:  [403 402]

# ==> Alignment :  1,2,3,4,0,5::4,5,3,1,2,0
# dc10b3f893746f3216144b7fe02dc16b
sbatch -J $exp_name $submit_to python 1.train.py --seed=1982 -f=$exp_name/2_randomized/ --train_bodies=403,402 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,2,3,4,0,5::4,5,3,1,2,0 --custom_align_max_joints=6

# train on bodies:  [508 514]

# ==> Alignment :  6,7,5,1,3,0,2,4::4,5,1,6,7,3,2,0
# 4dfdb9e8805f2c4ddb00d168f3f7ee9f
sbatch -J $exp_name $submit_to python 1.train.py --seed=2761 -f=$exp_name/2_randomized/ --train_bodies=508,514 --topology_wrapper=CustomAlignWrapper --custom_alignment=6,7,5,1,3,0,2,4::4,5,1,6,7,3,2,0 --custom_align_max_joints=8

# train on bodies:  [515 512]

# ==> Alignment :  5,2,3,4,7,6,0,1::7,5,3,4,0,6,1,2
# 5a7c88362063aa8f476114d32ee88cb6
sbatch -J $exp_name $submit_to python 1.train.py --seed=8419 -f=$exp_name/2_randomized/ --train_bodies=515,512 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,2,3,4,7,6,0,1::7,5,3,4,0,6,1,2 --custom_align_max_joints=8

# train on bodies:  [500 506]

# ==> Alignment :  6,1,3,7,2,4,5,0::6,5,0,7,1,3,4,2
# 2f26e79bfc99daf7c2067715a1a71ead
sbatch -J $exp_name $submit_to python 1.train.py --seed=4714 -f=$exp_name/2_randomized/ --train_bodies=500,506 --topology_wrapper=CustomAlignWrapper --custom_alignment=6,1,3,7,2,4,5,0::6,5,0,7,1,3,4,2 --custom_align_max_joints=8

# train on bodies:  [502 513]

# ==> Alignment :  2,4,3,6,1,0,5,7::3,2,6,1,5,0,4,7
# f93b07ab4ffbfe8d3351176908b60580
sbatch -J $exp_name $submit_to python 1.train.py --seed=8527 -f=$exp_name/2_randomized/ --train_bodies=502,513 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,3,6,1,0,5,7::3,2,6,1,5,0,4,7 --custom_align_max_joints=8

# train on bodies:  [515 507]

# ==> Alignment :  7,3,6,4,2,1,0,5::3,4,6,2,5,1,7,0
# 616646c10ae40d4eec0be77662d04f37
sbatch -J $exp_name $submit_to python 1.train.py --seed=9040 -f=$exp_name/2_randomized/ --train_bodies=515,507 --topology_wrapper=CustomAlignWrapper --custom_alignment=7,3,6,4,2,1,0,5::3,4,6,2,5,1,7,0 --custom_align_max_joints=8

# train on bodies:  [514 507]

# ==> Alignment :  7,6,3,1,5,0,4,2::0,2,5,3,4,1,7,6
# b8fe3d0cc0a76b9f0d90f5ea1669c0e4
sbatch -J $exp_name $submit_to python 1.train.py --seed=6462 -f=$exp_name/2_randomized/ --train_bodies=514,507 --topology_wrapper=CustomAlignWrapper --custom_alignment=7,6,3,1,5,0,4,2::0,2,5,3,4,1,7,6 --custom_align_max_joints=8

# train on bodies:  [515 510]

# ==> Alignment :  2,0,4,6,5,7,1,3::0,7,6,2,4,3,1,5
# 174c5ee8835233c5584ec585cd849575
sbatch -J $exp_name $submit_to python 1.train.py --seed=5401 -f=$exp_name/2_randomized/ --train_bodies=515,510 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,4,6,5,7,1,3::0,7,6,2,4,3,1,5 --custom_align_max_joints=8

# train on bodies:  [506 504]

# ==> Alignment :  1,2,3,0,6,4,7,5::6,1,0,3,7,4,2,5
# b0b6a9b6ad7921001b70ff2e686d1f55
sbatch -J $exp_name $submit_to python 1.train.py --seed=7243 -f=$exp_name/2_randomized/ --train_bodies=506,504 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,2,3,0,6,4,7,5::6,1,0,3,7,4,2,5 --custom_align_max_joints=8

# train on bodies:  [515 510]

# ==> Alignment :  5,6,1,2,0,4,3,7::7,4,0,2,1,3,6,5
# 99b3ed51848df57fa86d007ee0149ca0
sbatch -J $exp_name $submit_to python 1.train.py --seed=4122 -f=$exp_name/2_randomized/ --train_bodies=515,510 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,6,1,2,0,4,3,7::7,4,0,2,1,3,6,5 --custom_align_max_joints=8

# train on bodies:  [513 510]

# ==> Alignment :  7,3,2,4,6,1,5,0::3,5,2,6,0,7,1,4
# 5fad58478a44dadc69bef7a670b843f1
sbatch -J $exp_name $submit_to python 1.train.py --seed=2543 -f=$exp_name/2_randomized/ --train_bodies=513,510 --topology_wrapper=CustomAlignWrapper --custom_alignment=7,3,2,4,6,1,5,0::3,5,2,6,0,7,1,4 --custom_align_max_joints=8

# train on bodies:  [608 610]

# ==> Alignment :  0,2,1::2,1,0
# aa48a76501a6e1baa8d32bde12f549bd
sbatch -J $exp_name $submit_to python 1.train.py --seed=9412 -f=$exp_name/2_randomized/ --train_bodies=608,610 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::2,1,0 --custom_align_max_joints=3

# train on bodies:  [613 615]

# ==> Alignment :  1,2,0::0,2,1
# c3f8edc7af7ff0e9ff0fe74bd83a493e
sbatch -J $exp_name $submit_to python 1.train.py --seed=537 -f=$exp_name/2_randomized/ --train_bodies=613,615 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,2,0::0,2,1 --custom_align_max_joints=3

# train on bodies:  [609 611]

# ==> Alignment :  0,2,1::0,1,2
# 0a8ae5e55552c3ece67daeeb1befd854
sbatch -J $exp_name $submit_to python 1.train.py --seed=5699 -f=$exp_name/2_randomized/ --train_bodies=609,611 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::0,1,2 --custom_align_max_joints=3

# train on bodies:  [615 605]

# ==> Alignment :  1,0,2::2,0,1
# 9be3238214410427faa71decc11f0a98
sbatch -J $exp_name $submit_to python 1.train.py --seed=4307 -f=$exp_name/2_randomized/ --train_bodies=615,605 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,0,2::2,0,1 --custom_align_max_joints=3

# train on bodies:  [613 601]

# ==> Alignment :  1,0,2::2,1,0
# a372c44e124838a147fece5e74a8fa27
sbatch -J $exp_name $submit_to python 1.train.py --seed=919 -f=$exp_name/2_randomized/ --train_bodies=613,601 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,0,2::2,1,0 --custom_align_max_joints=3

# train on bodies:  [609 600]

# ==> Alignment :  1,2,0::0,1,2
# 4171f77b4285734653cb6c8a1fa4edbb
sbatch -J $exp_name $submit_to python 1.train.py --seed=1372 -f=$exp_name/2_randomized/ --train_bodies=609,600 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,2,0::0,1,2 --custom_align_max_joints=3

# train on bodies:  [604 611]

# ==> Alignment :  2,0,1::1,0,2
# f30e2d388b256aa8e66323c037d3702d
sbatch -J $exp_name $submit_to python 1.train.py --seed=7566 -f=$exp_name/2_randomized/ --train_bodies=604,611 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,1::1,0,2 --custom_align_max_joints=3

# train on bodies:  [607 612]

# ==> Alignment :  0,2,1::1,2,0
# a98a22db98dd94bf862256588e42b4e8
sbatch -J $exp_name $submit_to python 1.train.py --seed=2583 -f=$exp_name/2_randomized/ --train_bodies=607,612 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::1,2,0 --custom_align_max_joints=3

# train on bodies:  [605 604]

# ==> Alignment :  1,0,2::0,1,2
# b61e385cfca42ac2ef519f93b5025b0b
sbatch -J $exp_name $submit_to python 1.train.py --seed=4441 -f=$exp_name/2_randomized/ --train_bodies=605,604 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,0,2::0,1,2 --custom_align_max_joints=3

# train on bodies:  [605 612]

# ==> Alignment :  0,1,2::2,1,0
# c37f295f5cd36f5756581c401768783e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3239 -f=$exp_name/2_randomized/ --train_bodies=605,612 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,1,2::2,1,0 --custom_align_max_joints=3

# train on bodies:  [313 306 303 300]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4547 -f=$exp_name/4_aligned/ --train_bodies=313,306,303,300 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [313 305 300 314]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9556 -f=$exp_name/4_aligned/ --train_bodies=313,305,300,314 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [308 300 312 310]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2033 -f=$exp_name/4_aligned/ --train_bodies=308,300,312,310 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [301 305 312 306]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2181 -f=$exp_name/4_aligned/ --train_bodies=301,305,312,306 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [313 308 311 305]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6995 -f=$exp_name/4_aligned/ --train_bodies=313,308,311,305 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [302 303 313 309]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5480 -f=$exp_name/4_aligned/ --train_bodies=302,303,313,309 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [315 306 314 308]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2096 -f=$exp_name/4_aligned/ --train_bodies=315,306,314,308 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [303 306 308 302]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4211 -f=$exp_name/4_aligned/ --train_bodies=303,306,308,302 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [307 310 304 314]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=458 -f=$exp_name/4_aligned/ --train_bodies=307,310,304,314 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [314 303 309 302]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8812 -f=$exp_name/4_aligned/ --train_bodies=314,303,309,302 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [402 400 414 408]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8574 -f=$exp_name/4_aligned/ --train_bodies=402,400,414,408 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [404 401 407 410]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4444 -f=$exp_name/4_aligned/ --train_bodies=404,401,407,410 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [413 414 408 415]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6782 -f=$exp_name/4_aligned/ --train_bodies=413,414,408,415 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [409 410 412 411]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=501 -f=$exp_name/4_aligned/ --train_bodies=409,410,412,411 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [414 415 406 403]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6200 -f=$exp_name/4_aligned/ --train_bodies=414,415,406,403 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [408 401 413 411]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9979 -f=$exp_name/4_aligned/ --train_bodies=408,401,413,411 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [402 405 401 412]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5014 -f=$exp_name/4_aligned/ --train_bodies=402,405,401,412 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [404 414 408 401]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9341 -f=$exp_name/4_aligned/ --train_bodies=404,414,408,401 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [410 406 415 405]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4673 -f=$exp_name/4_aligned/ --train_bodies=410,406,415,405 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [410 400 407 404]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4532 -f=$exp_name/4_aligned/ --train_bodies=410,400,407,404 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [500 509 504 505]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1289 -f=$exp_name/4_aligned/ --train_bodies=500,509,504,505 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [511 508 510 501]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7293 -f=$exp_name/4_aligned/ --train_bodies=511,508,510,501 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [507 515 514 505]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1344 -f=$exp_name/4_aligned/ --train_bodies=507,515,514,505 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [512 503 504 515]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7291 -f=$exp_name/4_aligned/ --train_bodies=512,503,504,515 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [507 511 512 506]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9372 -f=$exp_name/4_aligned/ --train_bodies=507,511,512,506 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [504 506 501 500]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4829 -f=$exp_name/4_aligned/ --train_bodies=504,506,501,500 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [504 509 513 511]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1520 -f=$exp_name/4_aligned/ --train_bodies=504,509,513,511 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [507 515 511 510]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9224 -f=$exp_name/4_aligned/ --train_bodies=507,515,511,510 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [513 501 509 508]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9289 -f=$exp_name/4_aligned/ --train_bodies=513,501,509,508 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [515 503 500 512]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6400 -f=$exp_name/4_aligned/ --train_bodies=515,503,500,512 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [609 605 607 604]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3775 -f=$exp_name/4_aligned/ --train_bodies=609,605,607,604 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [605 607 612 609]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5200 -f=$exp_name/4_aligned/ --train_bodies=605,607,612,609 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [613 606 610 607]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7259 -f=$exp_name/4_aligned/ --train_bodies=613,606,610,607 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [609 613 603 607]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4023 -f=$exp_name/4_aligned/ --train_bodies=609,613,603,607 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [615 608 603 607]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1293 -f=$exp_name/4_aligned/ --train_bodies=615,608,603,607 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [610 609 604 613]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8524 -f=$exp_name/4_aligned/ --train_bodies=610,609,604,613 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [610 607 615 613]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=583 -f=$exp_name/4_aligned/ --train_bodies=610,607,615,613 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [606 610 614 615]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3864 -f=$exp_name/4_aligned/ --train_bodies=606,610,614,615 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [614 600 613 608]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6765 -f=$exp_name/4_aligned/ --train_bodies=614,600,613,608 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [600 601 607 611]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5724 -f=$exp_name/4_aligned/ --train_bodies=600,601,607,611 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [307 314 315 304]

# ==> Alignment :  5,1,4,2,3,0::0,2,5,4,1,3::1,0,4,2,3,5::4,5,2,0,3,1
# 6c826f4a3b2777199847da9b0520e59c
sbatch -J $exp_name $submit_to python 1.train.py --seed=5787 -f=$exp_name/4_randomized/ --train_bodies=307,314,315,304 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,1,4,2,3,0::0,2,5,4,1,3::1,0,4,2,3,5::4,5,2,0,3,1 --custom_align_max_joints=6

# train on bodies:  [310 315 307 303]

# ==> Alignment :  0,1,5,2,3,4::3,2,1,4,0,5::3,2,5,4,0,1::1,2,4,3,5,0
# 3041d45a98089c240cf7869636f374b3
sbatch -J $exp_name $submit_to python 1.train.py --seed=9606 -f=$exp_name/4_randomized/ --train_bodies=310,315,307,303 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,1,5,2,3,4::3,2,1,4,0,5::3,2,5,4,0,1::1,2,4,3,5,0 --custom_align_max_joints=6

# train on bodies:  [314 315 300 306]

# ==> Alignment :  0,5,4,3,1,2::4,1,0,5,2,3::3,1,2,4,5,0::1,2,3,5,0,4
# 6eccf905a447049d5d11c71db9e1ea06
sbatch -J $exp_name $submit_to python 1.train.py --seed=3325 -f=$exp_name/4_randomized/ --train_bodies=314,315,300,306 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,5,4,3,1,2::4,1,0,5,2,3::3,1,2,4,5,0::1,2,3,5,0,4 --custom_align_max_joints=6

# train on bodies:  [308 304 309 307]

# ==> Alignment :  0,1,2,4,5,3::0,3,5,1,2,4::2,4,5,1,0,3::3,1,0,4,5,2
# d6437936b8db438001c5906f3ada10d1
sbatch -J $exp_name $submit_to python 1.train.py --seed=7409 -f=$exp_name/4_randomized/ --train_bodies=308,304,309,307 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,1,2,4,5,3::0,3,5,1,2,4::2,4,5,1,0,3::3,1,0,4,5,2 --custom_align_max_joints=6

# train on bodies:  [302 308 303 306]

# ==> Alignment :  4,3,0,5,1,2::5,0,1,2,3,4::1,4,3,2,0,5::0,5,2,1,4,3
# 0bdbad9558caaeb69a38b7995a013c9a
sbatch -J $exp_name $submit_to python 1.train.py --seed=3714 -f=$exp_name/4_randomized/ --train_bodies=302,308,303,306 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,3,0,5,1,2::5,0,1,2,3,4::1,4,3,2,0,5::0,5,2,1,4,3 --custom_align_max_joints=6

# train on bodies:  [312 303 314 300]

# ==> Alignment :  4,3,0,2,1,5::3,5,4,2,0,1::0,2,1,3,5,4::3,4,0,5,1,2
# 4262b9107716caafd9ea3eb56a6db77f
sbatch -J $exp_name $submit_to python 1.train.py --seed=9475 -f=$exp_name/4_randomized/ --train_bodies=312,303,314,300 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,3,0,2,1,5::3,5,4,2,0,1::0,2,1,3,5,4::3,4,0,5,1,2 --custom_align_max_joints=6

# train on bodies:  [311 303 308 312]

# ==> Alignment :  3,4,5,1,2,0::5,3,0,1,2,4::2,1,4,3,5,0::5,0,3,4,2,1
# 760fb75422095446572d14dcec0e40c3
sbatch -J $exp_name $submit_to python 1.train.py --seed=278 -f=$exp_name/4_randomized/ --train_bodies=311,303,308,312 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,4,5,1,2,0::5,3,0,1,2,4::2,1,4,3,5,0::5,0,3,4,2,1 --custom_align_max_joints=6

# train on bodies:  [315 308 304 305]

# ==> Alignment :  2,4,0,5,1,3::5,4,1,3,0,2::0,1,5,2,3,4::5,3,1,2,4,0
# 775626df20aeb1655eea15520fb559a2
sbatch -J $exp_name $submit_to python 1.train.py --seed=8241 -f=$exp_name/4_randomized/ --train_bodies=315,308,304,305 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,0,5,1,3::5,4,1,3,0,2::0,1,5,2,3,4::5,3,1,2,4,0 --custom_align_max_joints=6

# train on bodies:  [315 306 314 305]

# ==> Alignment :  3,5,2,1,4,0::0,5,3,2,1,4::0,4,1,5,3,2::1,0,2,5,3,4
# 9c481dc81f516d0c55bcbeafc7ce3b81
sbatch -J $exp_name $submit_to python 1.train.py --seed=3725 -f=$exp_name/4_randomized/ --train_bodies=315,306,314,305 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,5,2,1,4,0::0,5,3,2,1,4::0,4,1,5,3,2::1,0,2,5,3,4 --custom_align_max_joints=6

# train on bodies:  [313 301 302 306]

# ==> Alignment :  1,3,5,2,4,0::4,5,2,3,1,0::0,2,4,5,3,1::1,0,5,3,2,4
# 756e734e3f4666e021da92dcbdf2a84b
sbatch -J $exp_name $submit_to python 1.train.py --seed=4569 -f=$exp_name/4_randomized/ --train_bodies=313,301,302,306 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,3,5,2,4,0::4,5,2,3,1,0::0,2,4,5,3,1::1,0,5,3,2,4 --custom_align_max_joints=6

# train on bodies:  [406 402 407 410]

# ==> Alignment :  3,2,4,5,1,0::1,4,5,2,3,0::2,3,5,0,4,1::5,0,1,3,4,2
# 059dd594bea4ee3cc2e3692de25213fc
sbatch -J $exp_name $submit_to python 1.train.py --seed=338 -f=$exp_name/4_randomized/ --train_bodies=406,402,407,410 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,2,4,5,1,0::1,4,5,2,3,0::2,3,5,0,4,1::5,0,1,3,4,2 --custom_align_max_joints=6

# train on bodies:  [406 401 415 407]

# ==> Alignment :  5,0,2,4,1,3::5,4,3,2,1,0::0,3,5,1,2,4::0,3,2,1,5,4
# b1a838b4ba0c2b8f8fbf71c438fbc6f6
sbatch -J $exp_name $submit_to python 1.train.py --seed=74 -f=$exp_name/4_randomized/ --train_bodies=406,401,415,407 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,0,2,4,1,3::5,4,3,2,1,0::0,3,5,1,2,4::0,3,2,1,5,4 --custom_align_max_joints=6

# train on bodies:  [400 409 413 414]

# ==> Alignment :  4,3,0,2,1,5::4,5,3,0,2,1::3,5,2,1,0,4::2,3,0,5,4,1
# dd725a0ec400237297637ce2698014c5
sbatch -J $exp_name $submit_to python 1.train.py --seed=7696 -f=$exp_name/4_randomized/ --train_bodies=400,409,413,414 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,3,0,2,1,5::4,5,3,0,2,1::3,5,2,1,0,4::2,3,0,5,4,1 --custom_align_max_joints=6

# train on bodies:  [402 401 404 410]

# ==> Alignment :  4,1,2,5,0,3::0,2,1,5,4,3::3,5,1,2,4,0::5,2,3,1,0,4
# 9ad8ae96a793110d1b40a5f0d5c61a83
sbatch -J $exp_name $submit_to python 1.train.py --seed=866 -f=$exp_name/4_randomized/ --train_bodies=402,401,404,410 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,1,2,5,0,3::0,2,1,5,4,3::3,5,1,2,4,0::5,2,3,1,0,4 --custom_align_max_joints=6

# train on bodies:  [412 410 406 408]

# ==> Alignment :  0,2,3,5,4,1::5,3,2,0,4,1::4,3,1,5,2,0::5,2,1,4,3,0
# 1b88d0de69bd99e70fb4436a730fb8c4
sbatch -J $exp_name $submit_to python 1.train.py --seed=5876 -f=$exp_name/4_randomized/ --train_bodies=412,410,406,408 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,3,5,4,1::5,3,2,0,4,1::4,3,1,5,2,0::5,2,1,4,3,0 --custom_align_max_joints=6

# train on bodies:  [410 409 404 402]

# ==> Alignment :  3,2,1,5,0,4::3,0,5,4,2,1::0,4,2,1,5,3::1,3,2,0,5,4
# 972a05392de0ab7d0456897d1a6c3c60
sbatch -J $exp_name $submit_to python 1.train.py --seed=153 -f=$exp_name/4_randomized/ --train_bodies=410,409,404,402 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,2,1,5,0,4::3,0,5,4,2,1::0,4,2,1,5,3::1,3,2,0,5,4 --custom_align_max_joints=6

# train on bodies:  [414 409 413 412]

# ==> Alignment :  5,0,2,3,4,1::0,4,5,2,3,1::2,5,1,3,4,0::4,0,3,2,5,1
# 84c98f37a9f5919293ad5691f45406d0
sbatch -J $exp_name $submit_to python 1.train.py --seed=8940 -f=$exp_name/4_randomized/ --train_bodies=414,409,413,412 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,0,2,3,4,1::0,4,5,2,3,1::2,5,1,3,4,0::4,0,3,2,5,1 --custom_align_max_joints=6

# train on bodies:  [406 413 408 409]

# ==> Alignment :  0,5,2,1,3,4::1,4,0,3,5,2::5,1,0,2,3,4::1,5,0,4,2,3
# 3dcb5c3b2028e82995e8e26b240c92cc
sbatch -J $exp_name $submit_to python 1.train.py --seed=4026 -f=$exp_name/4_randomized/ --train_bodies=406,413,408,409 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,5,2,1,3,4::1,4,0,3,5,2::5,1,0,2,3,4::1,5,0,4,2,3 --custom_align_max_joints=6

# train on bodies:  [410 401 408 411]

# ==> Alignment :  2,4,1,0,3,5::1,2,4,5,3,0::4,1,2,3,0,5::2,3,0,5,4,1
# f8d5be5a6262c8af15ddaf0607e200b3
sbatch -J $exp_name $submit_to python 1.train.py --seed=9114 -f=$exp_name/4_randomized/ --train_bodies=410,401,408,411 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,1,0,3,5::1,2,4,5,3,0::4,1,2,3,0,5::2,3,0,5,4,1 --custom_align_max_joints=6

# train on bodies:  [413 414 408 415]

# ==> Alignment :  1,3,4,5,2,0::1,5,3,0,4,2::3,4,0,2,1,5::0,2,4,1,5,3
# 08e026fb8ba7e6ff5d45ddb93d120c2b
sbatch -J $exp_name $submit_to python 1.train.py --seed=6782 -f=$exp_name/4_randomized/ --train_bodies=413,414,408,415 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,3,4,5,2,0::1,5,3,0,4,2::3,4,0,2,1,5::0,2,4,1,5,3 --custom_align_max_joints=6

# train on bodies:  [506 504 508 510]

# ==> Alignment :  0,3,4,2,7,1,5,6::6,3,1,7,4,5,2,0::1,2,5,0,7,4,6,3::2,7,0,1,3,6,5,4
# 885162df40dee292bf40c2931a5dc2ad
sbatch -J $exp_name $submit_to python 1.train.py --seed=2667 -f=$exp_name/4_randomized/ --train_bodies=506,504,508,510 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,3,4,2,7,1,5,6::6,3,1,7,4,5,2,0::1,2,5,0,7,4,6,3::2,7,0,1,3,6,5,4 --custom_align_max_joints=8

# train on bodies:  [508 507 503 501]

# ==> Alignment :  6,5,0,1,3,4,7,2::5,1,6,3,7,4,0,2::7,5,3,2,6,0,1,4::3,1,6,4,7,0,5,2
# 18669e2deac42182ceb49052744222e5
sbatch -J $exp_name $submit_to python 1.train.py --seed=9484 -f=$exp_name/4_randomized/ --train_bodies=508,507,503,501 --topology_wrapper=CustomAlignWrapper --custom_alignment=6,5,0,1,3,4,7,2::5,1,6,3,7,4,0,2::7,5,3,2,6,0,1,4::3,1,6,4,7,0,5,2 --custom_align_max_joints=8

# train on bodies:  [511 505 506 504]

# ==> Alignment :  2,4,6,7,1,0,5,3::3,5,1,6,7,2,4,0::4,5,1,7,3,6,0,2::7,6,3,4,2,1,5,0
# f42eca8b8e4eb47511c5912d0d259519
sbatch -J $exp_name $submit_to python 1.train.py --seed=2454 -f=$exp_name/4_randomized/ --train_bodies=511,505,506,504 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,6,7,1,0,5,3::3,5,1,6,7,2,4,0::4,5,1,7,3,6,0,2::7,6,3,4,2,1,5,0 --custom_align_max_joints=8

# train on bodies:  [504 502 515 503]

# ==> Alignment :  2,3,4,7,5,6,1,0::5,4,2,7,0,6,3,1::7,0,5,6,3,4,1,2::1,2,7,6,0,3,5,4
# 90d7355a022c08c767d7f61ce63fd460
sbatch -J $exp_name $submit_to python 1.train.py --seed=6471 -f=$exp_name/4_randomized/ --train_bodies=504,502,515,503 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,3,4,7,5,6,1,0::5,4,2,7,0,6,3,1::7,0,5,6,3,4,1,2::1,2,7,6,0,3,5,4 --custom_align_max_joints=8

# train on bodies:  [512 510 514 502]

# ==> Alignment :  0,2,5,6,4,7,3,1::6,4,5,3,0,1,7,2::7,3,5,0,1,2,6,4::6,5,3,0,1,4,2,7
# 9cc240f6184ff93d65cb5079dd84f509
sbatch -J $exp_name $submit_to python 1.train.py --seed=7526 -f=$exp_name/4_randomized/ --train_bodies=512,510,514,502 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,5,6,4,7,3,1::6,4,5,3,0,1,7,2::7,3,5,0,1,2,6,4::6,5,3,0,1,4,2,7 --custom_align_max_joints=8

# train on bodies:  [511 514 512 503]

# ==> Alignment :  5,3,7,1,2,4,6,0::1,2,7,6,0,3,5,4::3,1,5,0,2,7,4,6::5,2,7,0,1,3,6,4
# 5ef74da7e25383511d3ff6ed95f2b57c
sbatch -J $exp_name $submit_to python 1.train.py --seed=6812 -f=$exp_name/4_randomized/ --train_bodies=511,514,512,503 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,3,7,1,2,4,6,0::1,2,7,6,0,3,5,4::3,1,5,0,2,7,4,6::5,2,7,0,1,3,6,4 --custom_align_max_joints=8

# train on bodies:  [503 511 510 501]

# ==> Alignment :  0,3,2,7,5,4,1,6::2,7,3,0,4,1,5,6::7,5,2,0,4,6,1,3::3,4,7,2,0,6,1,5
# 192bcc46475785bd41d3373769c77edf
sbatch -J $exp_name $submit_to python 1.train.py --seed=1232 -f=$exp_name/4_randomized/ --train_bodies=503,511,510,501 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,3,2,7,5,4,1,6::2,7,3,0,4,1,5,6::7,5,2,0,4,6,1,3::3,4,7,2,0,6,1,5 --custom_align_max_joints=8

# train on bodies:  [514 506 508 510]

# ==> Alignment :  2,7,6,5,0,1,4,3::0,1,4,7,5,2,6,3::5,3,7,0,1,6,2,4::4,3,2,1,6,7,0,5
# a0bc4477dcf7a4730adf112c8fa356ef
sbatch -J $exp_name $submit_to python 1.train.py --seed=6717 -f=$exp_name/4_randomized/ --train_bodies=514,506,508,510 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,7,6,5,0,1,4,3::0,1,4,7,5,2,6,3::5,3,7,0,1,6,2,4::4,3,2,1,6,7,0,5 --custom_align_max_joints=8

# train on bodies:  [505 504 501 510]

# ==> Alignment :  1,3,5,4,0,2,7,6::2,0,3,6,7,5,1,4::0,1,5,3,6,4,2,7::1,7,6,0,2,4,5,3
# 12553eb3d75ee4af81c82a72f529017b
sbatch -J $exp_name $submit_to python 1.train.py --seed=7656 -f=$exp_name/4_randomized/ --train_bodies=505,504,501,510 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,3,5,4,0,2,7,6::2,0,3,6,7,5,1,4::0,1,5,3,6,4,2,7::1,7,6,0,2,4,5,3 --custom_align_max_joints=8

# train on bodies:  [515 507 506 511]

# ==> Alignment :  1,3,7,4,2,6,0,5::7,6,0,1,5,4,2,3::0,7,5,3,1,4,6,2::4,2,5,0,1,3,6,7
# 25bf10ac7a2b2f0c8239bc222f7e9240
sbatch -J $exp_name $submit_to python 1.train.py --seed=8045 -f=$exp_name/4_randomized/ --train_bodies=515,507,506,511 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,3,7,4,2,6,0,5::7,6,0,1,5,4,2,3::0,7,5,3,1,4,6,2::4,2,5,0,1,3,6,7 --custom_align_max_joints=8

# train on bodies:  [601 604 612 603]

# ==> Alignment :  2,1,0::1,0,2::0,2,1::0,2,1
# f241e2e1490c973e68f2f195c875e440
sbatch -J $exp_name $submit_to python 1.train.py --seed=7624 -f=$exp_name/4_randomized/ --train_bodies=601,604,612,603 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,1,0::1,0,2::0,2,1::0,2,1 --custom_align_max_joints=3

# train on bodies:  [614 613 611 601]

# ==> Alignment :  1,0,2::2,1,0::0,1,2::0,1,2
# b13855592f1e5c3cfd5867021c48c7ca
sbatch -J $exp_name $submit_to python 1.train.py --seed=3829 -f=$exp_name/4_randomized/ --train_bodies=614,613,611,601 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,0,2::2,1,0::0,1,2::0,1,2 --custom_align_max_joints=3

# train on bodies:  [606 602 610 614]

# ==> Alignment :  2,0,1::1,2,0::2,0,1::1,0,2
# 5ed49e48a2a0e4233a233bf3dc71ac44
sbatch -J $exp_name $submit_to python 1.train.py --seed=8076 -f=$exp_name/4_randomized/ --train_bodies=606,602,610,614 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,1::1,2,0::2,0,1::1,0,2 --custom_align_max_joints=3

# train on bodies:  [612 609 611 601]

# ==> Alignment :  0,1,2::1,0,2::0,1,2::0,2,1
# a051861147fd1375ea024f6a35fea82d
sbatch -J $exp_name $submit_to python 1.train.py --seed=2693 -f=$exp_name/4_randomized/ --train_bodies=612,609,611,601 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,1,2::1,0,2::0,1,2::0,2,1 --custom_align_max_joints=3

# train on bodies:  [615 607 602 604]

# ==> Alignment :  2,0,1::1,0,2::2,1,0::1,0,2
# 6701561582ed1ae299a7a739819b0692
sbatch -J $exp_name $submit_to python 1.train.py --seed=6528 -f=$exp_name/4_randomized/ --train_bodies=615,607,602,604 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,1::1,0,2::2,1,0::1,0,2 --custom_align_max_joints=3

# train on bodies:  [603 606 604 601]

# ==> Alignment :  0,2,1::0,1,2::0,1,2::1,0,2
# 27adb4c335e6e4a051d35a00bb8826a6
sbatch -J $exp_name $submit_to python 1.train.py --seed=2204 -f=$exp_name/4_randomized/ --train_bodies=603,606,604,601 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::0,1,2::0,1,2::1,0,2 --custom_align_max_joints=3

# train on bodies:  [615 604 606 614]

# ==> Alignment :  2,0,1::0,2,1::0,2,1::0,2,1
# 6f200cd555caa97b0f6cbf43433eee55
sbatch -J $exp_name $submit_to python 1.train.py --seed=2715 -f=$exp_name/4_randomized/ --train_bodies=615,604,606,614 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,1::0,2,1::0,2,1::0,2,1 --custom_align_max_joints=3

# train on bodies:  [608 611 602 610]

# ==> Alignment :  2,1,0::1,2,0::1,0,2::2,0,1
# af3f6488dfeb158277ebf27cab5aaa69
sbatch -J $exp_name $submit_to python 1.train.py --seed=6229 -f=$exp_name/4_randomized/ --train_bodies=608,611,602,610 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,1,0::1,2,0::1,0,2::2,0,1 --custom_align_max_joints=3

# train on bodies:  [601 612 608 611]

# ==> Alignment :  1,2,0::0,2,1::1,0,2::0,2,1
# fd0cd977022a9c3b6bcaf0d2d45989c2
sbatch -J $exp_name $submit_to python 1.train.py --seed=4726 -f=$exp_name/4_randomized/ --train_bodies=601,612,608,611 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,2,0::0,2,1::1,0,2::0,2,1 --custom_align_max_joints=3

# train on bodies:  [615 607 604 611]

# ==> Alignment :  0,2,1::2,0,1::2,1,0::2,0,1
# 37a9d0a222e1c429f3921d0095416a9e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8413 -f=$exp_name/4_randomized/ --train_bodies=615,607,604,611 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::2,0,1::2,1,0::2,0,1 --custom_align_max_joints=3

# train on bodies:  [306 314 305 315 304 311 313 301]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6825 -f=$exp_name/8_aligned/ --train_bodies=306,314,305,315,304,311,313,301 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [314 310 306 308 300 305 311 304]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2169 -f=$exp_name/8_aligned/ --train_bodies=314,310,306,308,300,305,311,304 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [300 314 302 309 305 304 306 303]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=581 -f=$exp_name/8_aligned/ --train_bodies=300,314,302,309,305,304,306,303 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [311 310 303 312 313 307 304 308]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1345 -f=$exp_name/8_aligned/ --train_bodies=311,310,303,312,313,307,304,308 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [309 301 312 311 314 305 302 307]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=635 -f=$exp_name/8_aligned/ --train_bodies=309,301,312,311,314,305,302,307 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [313 314 300 307 310 301 311 302]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4548 -f=$exp_name/8_aligned/ --train_bodies=313,314,300,307,310,301,311,302 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [301 312 304 308 305 313 307 300]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=381 -f=$exp_name/8_aligned/ --train_bodies=301,312,304,308,305,313,307,300 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [305 303 309 315 308 313 301 300]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=772 -f=$exp_name/8_aligned/ --train_bodies=305,303,309,315,308,313,301,300 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [304 308 306 313 305 314 311 312]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5417 -f=$exp_name/8_aligned/ --train_bodies=304,308,306,313,305,314,311,312 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [308 315 314 307 311 306 300 303]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5600 -f=$exp_name/8_aligned/ --train_bodies=308,315,314,307,311,306,300,303 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [406 415 404 402 409 403 413 412]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2191 -f=$exp_name/8_aligned/ --train_bodies=406,415,404,402,409,403,413,412 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [408 414 412 405 409 410 407 402]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=406 -f=$exp_name/8_aligned/ --train_bodies=408,414,412,405,409,410,407,402 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [415 408 406 409 410 402 411 404]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9529 -f=$exp_name/8_aligned/ --train_bodies=415,408,406,409,410,402,411,404 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [411 400 401 405 404 408 402 407]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6061 -f=$exp_name/8_aligned/ --train_bodies=411,400,401,405,404,408,402,407 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [402 414 404 407 412 405 408 406]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8470 -f=$exp_name/8_aligned/ --train_bodies=402,414,404,407,412,405,408,406 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [403 415 400 407 413 406 414 402]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9247 -f=$exp_name/8_aligned/ --train_bodies=403,415,400,407,413,406,414,402 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [414 408 402 412 407 410 406 400]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3623 -f=$exp_name/8_aligned/ --train_bodies=414,408,402,412,407,410,406,400 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [404 402 409 407 400 405 413 401]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4564 -f=$exp_name/8_aligned/ --train_bodies=404,402,409,407,400,405,413,401 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [415 407 414 404 410 409 403 413]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2945 -f=$exp_name/8_aligned/ --train_bodies=415,407,414,404,410,409,403,413 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [412 409 415 402 404 407 413 403]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1774 -f=$exp_name/8_aligned/ --train_bodies=412,409,415,402,404,407,413,403 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [508 511 507 514 509 506 512 504]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9336 -f=$exp_name/8_aligned/ --train_bodies=508,511,507,514,509,506,512,504 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [503 515 504 505 513 512 500 511]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2885 -f=$exp_name/8_aligned/ --train_bodies=503,515,504,505,513,512,500,511 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [501 502 510 512 500 515 511 509]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1726 -f=$exp_name/8_aligned/ --train_bodies=501,502,510,512,500,515,511,509 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [515 508 501 513 506 503 507 510]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5294 -f=$exp_name/8_aligned/ --train_bodies=515,508,501,513,506,503,507,510 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [506 511 507 508 510 513 514 505]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2290 -f=$exp_name/8_aligned/ --train_bodies=506,511,507,508,510,513,514,505 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [505 514 508 501 506 503 509 512]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2312 -f=$exp_name/8_aligned/ --train_bodies=505,514,508,501,506,503,509,512 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [515 508 510 504 509 511 507 500]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6882 -f=$exp_name/8_aligned/ --train_bodies=515,508,510,504,509,511,507,500 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [506 501 507 502 504 508 503 505]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9105 -f=$exp_name/8_aligned/ --train_bodies=506,501,507,502,504,508,503,505 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [514 515 512 502 511 509 513 507]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=645 -f=$exp_name/8_aligned/ --train_bodies=514,515,512,502,511,509,513,507 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [502 503 513 501 514 504 511 507]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6378 -f=$exp_name/8_aligned/ --train_bodies=502,503,513,501,514,504,511,507 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [614 610 600 601 603 615 607 605]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=757 -f=$exp_name/8_aligned/ --train_bodies=614,610,600,601,603,615,607,605 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [608 611 604 614 612 607 602 610]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1378 -f=$exp_name/8_aligned/ --train_bodies=608,611,604,614,612,607,602,610 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [602 603 608 611 600 609 604 607]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5032 -f=$exp_name/8_aligned/ --train_bodies=602,603,608,611,600,609,604,607 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [606 601 610 605 607 603 614 604]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1043 -f=$exp_name/8_aligned/ --train_bodies=606,601,610,605,607,603,614,604 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [612 601 613 602 611 610 608 605]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6452 -f=$exp_name/8_aligned/ --train_bodies=612,601,613,602,611,610,608,605 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [605 609 610 606 608 613 603 614]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1354 -f=$exp_name/8_aligned/ --train_bodies=605,609,610,606,608,613,603,614 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [615 610 601 608 604 611 602 605]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=834 -f=$exp_name/8_aligned/ --train_bodies=615,610,601,608,604,611,602,605 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [615 601 610 613 600 608 612 609]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3973 -f=$exp_name/8_aligned/ --train_bodies=615,601,610,613,600,608,612,609 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [605 610 609 600 601 603 615 606]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=266 -f=$exp_name/8_aligned/ --train_bodies=605,610,609,600,601,603,615,606 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [604 610 611 601 603 600 606 605]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2279 -f=$exp_name/8_aligned/ --train_bodies=604,610,611,601,603,600,606,605 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [306 303 304 312 301 310 315 307]

# ==> Alignment :  5,2,1,4,3,0::0,2,4,5,1,3::0,5,1,4,2,3::1,0,4,5,2,3::0,5,3,2,1,4::1,5,2,4,3,0::5,1,4,3,0,2::0,5,3,2,1,4
# 6b0700eee74e493b00791da887f3d9e5
sbatch -J $exp_name $submit_to python 1.train.py --seed=4367 -f=$exp_name/8_randomized/ --train_bodies=306,303,304,312,301,310,315,307 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,2,1,4,3,0::0,2,4,5,1,3::0,5,1,4,2,3::1,0,4,5,2,3::0,5,3,2,1,4::1,5,2,4,3,0::5,1,4,3,0,2::0,5,3,2,1,4 --custom_align_max_joints=6

# train on bodies:  [301 303 309 308 314 306 313 302]

# ==> Alignment :  1,0,5,3,4,2::0,5,2,1,4,3::5,1,0,4,3,2::2,0,1,4,5,3::2,1,0,5,4,3::4,1,5,0,3,2::2,3,4,5,1,0::1,3,2,0,5,4
# 4d8522e63ec788dd2608320b54f3289f
sbatch -J $exp_name $submit_to python 1.train.py --seed=7391 -f=$exp_name/8_randomized/ --train_bodies=301,303,309,308,314,306,313,302 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,0,5,3,4,2::0,5,2,1,4,3::5,1,0,4,3,2::2,0,1,4,5,3::2,1,0,5,4,3::4,1,5,0,3,2::2,3,4,5,1,0::1,3,2,0,5,4 --custom_align_max_joints=6

# train on bodies:  [304 300 301 313 315 302 306 308]

# ==> Alignment :  0,2,5,3,1,4::0,2,1,3,5,4::4,2,3,1,0,5::5,4,2,3,1,0::0,3,4,2,1,5::5,2,0,1,3,4::1,2,5,4,0,3::3,1,5,2,0,4
# c50b0bc7010ceafbfa4a8316d911b76e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7068 -f=$exp_name/8_randomized/ --train_bodies=304,300,301,313,315,302,306,308 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,5,3,1,4::0,2,1,3,5,4::4,2,3,1,0,5::5,4,2,3,1,0::0,3,4,2,1,5::5,2,0,1,3,4::1,2,5,4,0,3::3,1,5,2,0,4 --custom_align_max_joints=6

# train on bodies:  [302 308 303 300 315 301 307 311]

# ==> Alignment :  2,3,1,0,4,5::3,2,1,5,4,0::5,1,0,3,4,2::4,1,0,3,2,5::3,2,1,4,0,5::3,2,4,1,0,5::1,0,2,4,3,5::0,1,4,3,5,2
# 1ed28a62e12c602c1f2b90fd62a9221f
sbatch -J $exp_name $submit_to python 1.train.py --seed=9620 -f=$exp_name/8_randomized/ --train_bodies=302,308,303,300,315,301,307,311 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,3,1,0,4,5::3,2,1,5,4,0::5,1,0,3,4,2::4,1,0,3,2,5::3,2,1,4,0,5::3,2,4,1,0,5::1,0,2,4,3,5::0,1,4,3,5,2 --custom_align_max_joints=6

# train on bodies:  [308 312 314 306 305 304 311 300]

# ==> Alignment :  0,3,4,2,5,1::3,1,2,5,0,4::5,2,4,1,0,3::1,4,2,5,3,0::1,5,0,4,3,2::3,1,2,4,5,0::5,3,1,2,0,4::4,2,5,1,3,0
# 6de6f351df602ab755fbfee2d61314da
sbatch -J $exp_name $submit_to python 1.train.py --seed=3915 -f=$exp_name/8_randomized/ --train_bodies=308,312,314,306,305,304,311,300 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,3,4,2,5,1::3,1,2,5,0,4::5,2,4,1,0,3::1,4,2,5,3,0::1,5,0,4,3,2::3,1,2,4,5,0::5,3,1,2,0,4::4,2,5,1,3,0 --custom_align_max_joints=6

# train on bodies:  [314 311 308 304 307 310 300 309]

# ==> Alignment :  0,4,5,3,1,2::2,0,1,5,3,4::4,0,3,2,1,5::4,2,1,0,5,3::2,3,4,1,5,0::4,3,1,5,2,0::1,5,2,0,3,4::2,1,3,4,5,0
# 5e61e60a2c17fe4ea05e04c9c54a09de
sbatch -J $exp_name $submit_to python 1.train.py --seed=5910 -f=$exp_name/8_randomized/ --train_bodies=314,311,308,304,307,310,300,309 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,4,5,3,1,2::2,0,1,5,3,4::4,0,3,2,1,5::4,2,1,0,5,3::2,3,4,1,5,0::4,3,1,5,2,0::1,5,2,0,3,4::2,1,3,4,5,0 --custom_align_max_joints=6

# train on bodies:  [307 315 309 313 303 301 308 300]

# ==> Alignment :  1,0,3,5,2,4::3,1,4,0,5,2::2,5,0,1,4,3::2,4,3,1,5,0::1,0,3,5,2,4::2,4,0,3,1,5::5,2,0,4,1,3::5,4,2,0,3,1
# f8dc5d48672ceffea4caa285300ebec4
sbatch -J $exp_name $submit_to python 1.train.py --seed=1607 -f=$exp_name/8_randomized/ --train_bodies=307,315,309,313,303,301,308,300 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,0,3,5,2,4::3,1,4,0,5,2::2,5,0,1,4,3::2,4,3,1,5,0::1,0,3,5,2,4::2,4,0,3,1,5::5,2,0,4,1,3::5,4,2,0,3,1 --custom_align_max_joints=6

# train on bodies:  [305 303 315 310 304 302 307 301]

# ==> Alignment :  5,3,2,1,4,0::1,3,0,2,4,5::4,2,5,1,0,3::1,0,4,5,3,2::5,0,4,2,3,1::0,4,3,5,1,2::2,5,1,4,0,3::4,2,1,0,5,3
# 30821c3fa6d9312d2cd584f4365a271c
sbatch -J $exp_name $submit_to python 1.train.py --seed=3234 -f=$exp_name/8_randomized/ --train_bodies=305,303,315,310,304,302,307,301 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,3,2,1,4,0::1,3,0,2,4,5::4,2,5,1,0,3::1,0,4,5,3,2::5,0,4,2,3,1::0,4,3,5,1,2::2,5,1,4,0,3::4,2,1,0,5,3 --custom_align_max_joints=6

# train on bodies:  [306 307 302 310 308 304 313 309]

# ==> Alignment :  4,1,2,0,5,3::3,2,1,5,4,0::1,3,5,2,4,0::4,0,1,3,2,5::2,3,4,1,5,0::4,2,1,5,3,0::5,4,3,0,1,2::4,3,2,5,1,0
# f9c1d368c039c46a8146e085dae3f8ad
sbatch -J $exp_name $submit_to python 1.train.py --seed=8744 -f=$exp_name/8_randomized/ --train_bodies=306,307,302,310,308,304,313,309 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,1,2,0,5,3::3,2,1,5,4,0::1,3,5,2,4,0::4,0,1,3,2,5::2,3,4,1,5,0::4,2,1,5,3,0::5,4,3,0,1,2::4,3,2,5,1,0 --custom_align_max_joints=6

# train on bodies:  [301 306 304 310 307 300 313 315]

# ==> Alignment :  0,4,1,5,2,3::2,4,3,1,5,0::4,3,5,2,1,0::2,1,3,0,5,4::1,4,0,2,3,5::4,5,1,3,0,2::5,3,4,2,0,1::0,3,1,2,5,4
# b48a8f6849da243ae9f5f191086aca25
sbatch -J $exp_name $submit_to python 1.train.py --seed=5589 -f=$exp_name/8_randomized/ --train_bodies=301,306,304,310,307,300,313,315 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,4,1,5,2,3::2,4,3,1,5,0::4,3,5,2,1,0::2,1,3,0,5,4::1,4,0,2,3,5::4,5,1,3,0,2::5,3,4,2,0,1::0,3,1,2,5,4 --custom_align_max_joints=6

# train on bodies:  [415 409 403 406 414 407 401 411]

# ==> Alignment :  1,5,3,0,4,2::2,0,5,4,1,3::5,3,4,1,0,2::5,3,0,1,4,2::4,2,0,5,3,1::0,5,4,1,3,2::0,5,4,3,2,1::1,4,5,0,3,2
# 41b8c1b424628dd92d3cdd90cbb0f0ec
sbatch -J $exp_name $submit_to python 1.train.py --seed=5327 -f=$exp_name/8_randomized/ --train_bodies=415,409,403,406,414,407,401,411 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,5,3,0,4,2::2,0,5,4,1,3::5,3,4,1,0,2::5,3,0,1,4,2::4,2,0,5,3,1::0,5,4,1,3,2::0,5,4,3,2,1::1,4,5,0,3,2 --custom_align_max_joints=6

# train on bodies:  [409 404 405 415 406 403 412 414]

# ==> Alignment :  1,3,0,4,2,5::2,5,4,0,1,3::5,3,0,2,1,4::0,2,3,1,5,4::4,0,1,2,5,3::1,3,4,2,5,0::5,4,0,3,2,1::1,5,0,3,2,4
# 1bf2bfb7423f4c6da503090027032f2d
sbatch -J $exp_name $submit_to python 1.train.py --seed=5944 -f=$exp_name/8_randomized/ --train_bodies=409,404,405,415,406,403,412,414 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,3,0,4,2,5::2,5,4,0,1,3::5,3,0,2,1,4::0,2,3,1,5,4::4,0,1,2,5,3::1,3,4,2,5,0::5,4,0,3,2,1::1,5,0,3,2,4 --custom_align_max_joints=6

# train on bodies:  [413 404 415 406 403 407 405 411]

# ==> Alignment :  1,3,5,0,4,2::4,1,2,0,5,3::4,2,5,0,3,1::4,1,0,3,2,5::4,0,2,3,1,5::2,4,1,0,5,3::5,1,4,0,3,2::4,1,2,0,3,5
# c875ff391bc088ebdd5fcb963b3c9dce
sbatch -J $exp_name $submit_to python 1.train.py --seed=8964 -f=$exp_name/8_randomized/ --train_bodies=413,404,415,406,403,407,405,411 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,3,5,0,4,2::4,1,2,0,5,3::4,2,5,0,3,1::4,1,0,3,2,5::4,0,2,3,1,5::2,4,1,0,5,3::5,1,4,0,3,2::4,1,2,0,3,5 --custom_align_max_joints=6

# train on bodies:  [407 414 413 405 409 412 402 408]

# ==> Alignment :  2,4,5,1,3,0::4,3,1,5,0,2::1,5,3,2,4,0::5,4,3,0,2,1::0,5,2,4,1,3::1,4,2,0,5,3::1,3,4,2,5,0::0,1,2,3,5,4
# f09dccaed62d12793e7a683fa6bec32c
sbatch -J $exp_name $submit_to python 1.train.py --seed=48 -f=$exp_name/8_randomized/ --train_bodies=407,414,413,405,409,412,402,408 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,5,1,3,0::4,3,1,5,0,2::1,5,3,2,4,0::5,4,3,0,2,1::0,5,2,4,1,3::1,4,2,0,5,3::1,3,4,2,5,0::0,1,2,3,5,4 --custom_align_max_joints=6

# train on bodies:  [408 411 407 414 409 406 412 404]

# ==> Alignment :  5,3,1,4,2,0::1,3,5,2,0,4::2,5,1,4,3,0::5,2,1,4,0,3::3,0,4,5,1,2::2,5,1,0,4,3::5,2,0,4,1,3::3,0,1,4,2,5
# da3dcb666b9ca201fa2f065b003fe0e0
sbatch -J $exp_name $submit_to python 1.train.py --seed=9336 -f=$exp_name/8_randomized/ --train_bodies=408,411,407,414,409,406,412,404 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,3,1,4,2,0::1,3,5,2,0,4::2,5,1,4,3,0::5,2,1,4,0,3::3,0,4,5,1,2::2,5,1,0,4,3::5,2,0,4,1,3::3,0,1,4,2,5 --custom_align_max_joints=6

# train on bodies:  [408 403 413 410 415 406 405 401]

# ==> Alignment :  2,4,5,3,1,0::2,3,5,1,4,0::5,1,4,3,2,0::3,2,1,4,0,5::1,2,0,3,5,4::4,0,3,1,2,5::1,3,5,4,2,0::2,4,5,1,3,0
# e62dc290c163aca43fd163e938eec4f2
sbatch -J $exp_name $submit_to python 1.train.py --seed=6844 -f=$exp_name/8_randomized/ --train_bodies=408,403,413,410,415,406,405,401 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,5,3,1,0::2,3,5,1,4,0::5,1,4,3,2,0::3,2,1,4,0,5::1,2,0,3,5,4::4,0,3,1,2,5::1,3,5,4,2,0::2,4,5,1,3,0 --custom_align_max_joints=6

# train on bodies:  [406 415 411 413 412 400 405 403]

# ==> Alignment :  3,5,0,2,4,1::0,2,1,3,4,5::0,2,5,1,3,4::3,1,5,2,4,0::0,2,3,1,4,5::2,4,1,3,5,0::5,2,3,0,4,1::2,5,0,3,4,1
# 6bbd1312962c6e4e54de5e708813bea7
sbatch -J $exp_name $submit_to python 1.train.py --seed=840 -f=$exp_name/8_randomized/ --train_bodies=406,415,411,413,412,400,405,403 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,5,0,2,4,1::0,2,1,3,4,5::0,2,5,1,3,4::3,1,5,2,4,0::0,2,3,1,4,5::2,4,1,3,5,0::5,2,3,0,4,1::2,5,0,3,4,1 --custom_align_max_joints=6

# train on bodies:  [410 414 404 415 411 408 407 412]

# ==> Alignment :  0,5,4,1,3,2::1,2,0,3,4,5::5,3,0,4,1,2::3,4,5,1,2,0::2,4,3,0,5,1::5,4,2,1,0,3::5,4,0,1,2,3::5,3,2,4,1,0
# a340f961b9e3dd151ef4189f15d15f2b
sbatch -J $exp_name $submit_to python 1.train.py --seed=1646 -f=$exp_name/8_randomized/ --train_bodies=410,414,404,415,411,408,407,412 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,5,4,1,3,2::1,2,0,3,4,5::5,3,0,4,1,2::3,4,5,1,2,0::2,4,3,0,5,1::5,4,2,1,0,3::5,4,0,1,2,3::5,3,2,4,1,0 --custom_align_max_joints=6

# train on bodies:  [415 402 409 408 401 400 405 407]

# ==> Alignment :  1,3,5,0,2,4::2,5,3,0,1,4::2,3,0,4,5,1::0,4,5,3,1,2::0,1,3,2,5,4::1,4,5,0,2,3::5,1,4,0,2,3::5,3,1,4,0,2
# 7c9ba30b8e3ed7ecefcc051ec99ba1eb
sbatch -J $exp_name $submit_to python 1.train.py --seed=8639 -f=$exp_name/8_randomized/ --train_bodies=415,402,409,408,401,400,405,407 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,3,5,0,2,4::2,5,3,0,1,4::2,3,0,4,5,1::0,4,5,3,1,2::0,1,3,2,5,4::1,4,5,0,2,3::5,1,4,0,2,3::5,3,1,4,0,2 --custom_align_max_joints=6

# train on bodies:  [400 411 415 402 413 407 412 410]

# ==> Alignment :  3,2,5,4,0,1::3,4,2,0,1,5::2,3,5,0,1,4::2,4,5,1,0,3::4,5,0,2,1,3::3,0,2,5,1,4::1,5,4,0,3,2::0,4,1,5,2,3
# 6da9538852c84712f8e0f9b2fcdfa898
sbatch -J $exp_name $submit_to python 1.train.py --seed=5548 -f=$exp_name/8_randomized/ --train_bodies=400,411,415,402,413,407,412,410 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,2,5,4,0,1::3,4,2,0,1,5::2,3,5,0,1,4::2,4,5,1,0,3::4,5,0,2,1,3::3,0,2,5,1,4::1,5,4,0,3,2::0,4,1,5,2,3 --custom_align_max_joints=6

# train on bodies:  [504 509 507 514 510 508 506 500]

# ==> Alignment :  3,1,0,6,7,4,5,2::1,5,3,2,0,6,4,7::7,6,2,0,1,5,4,3::2,4,6,7,1,0,5,3::1,0,5,6,7,4,3,2::0,6,2,1,3,4,5,7::0,6,2,3,4,1,7,5::0,1,5,7,4,3,2,6
# cf94e832484b3c7ae4e94e7225857bdb
sbatch -J $exp_name $submit_to python 1.train.py --seed=6276 -f=$exp_name/8_randomized/ --train_bodies=504,509,507,514,510,508,506,500 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,1,0,6,7,4,5,2::1,5,3,2,0,6,4,7::7,6,2,0,1,5,4,3::2,4,6,7,1,0,5,3::1,0,5,6,7,4,3,2::0,6,2,1,3,4,5,7::0,6,2,3,4,1,7,5::0,1,5,7,4,3,2,6 --custom_align_max_joints=8

# train on bodies:  [503 507 512 501 515 511 502 508]

# ==> Alignment :  7,4,3,1,5,0,6,2::1,6,4,5,7,3,2,0::4,3,7,6,5,1,0,2::5,0,7,1,2,6,4,3::2,0,6,7,4,3,5,1::3,2,0,6,4,5,1,7::7,0,1,5,4,2,6,3::4,5,7,6,0,1,3,2
# 771bca9c94664ec0fed4d79f69458ec3
sbatch -J $exp_name $submit_to python 1.train.py --seed=8548 -f=$exp_name/8_randomized/ --train_bodies=503,507,512,501,515,511,502,508 --topology_wrapper=CustomAlignWrapper --custom_alignment=7,4,3,1,5,0,6,2::1,6,4,5,7,3,2,0::4,3,7,6,5,1,0,2::5,0,7,1,2,6,4,3::2,0,6,7,4,3,5,1::3,2,0,6,4,5,1,7::7,0,1,5,4,2,6,3::4,5,7,6,0,1,3,2 --custom_align_max_joints=8

# train on bodies:  [502 510 505 509 501 513 503 511]

# ==> Alignment :  3,1,2,4,5,7,0,6::4,2,0,7,3,1,5,6::2,4,3,6,0,5,1,7::6,1,5,2,4,3,0,7::1,5,3,0,7,6,4,2::0,5,2,3,6,7,4,1::3,7,6,4,0,2,5,1::3,5,4,6,0,1,2,7
# 4fcd21ff3a7f83a7c853fe9c3679ace3
sbatch -J $exp_name $submit_to python 1.train.py --seed=5478 -f=$exp_name/8_randomized/ --train_bodies=502,510,505,509,501,513,503,511 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,1,2,4,5,7,0,6::4,2,0,7,3,1,5,6::2,4,3,6,0,5,1,7::6,1,5,2,4,3,0,7::1,5,3,0,7,6,4,2::0,5,2,3,6,7,4,1::3,7,6,4,0,2,5,1::3,5,4,6,0,1,2,7 --custom_align_max_joints=8

# train on bodies:  [505 514 500 506 515 510 504 501]

# ==> Alignment :  7,6,0,5,3,1,2,4::1,0,7,2,5,4,3,6::5,3,4,6,1,0,7,2::3,2,1,5,6,7,4,0::6,4,3,0,5,1,2,7::2,3,0,6,1,5,7,4::2,6,7,0,1,3,5,4::7,6,4,0,2,3,5,1
# 483b37e5bbfae474091027e503259ff9
sbatch -J $exp_name $submit_to python 1.train.py --seed=6646 -f=$exp_name/8_randomized/ --train_bodies=505,514,500,506,515,510,504,501 --topology_wrapper=CustomAlignWrapper --custom_alignment=7,6,0,5,3,1,2,4::1,0,7,2,5,4,3,6::5,3,4,6,1,0,7,2::3,2,1,5,6,7,4,0::6,4,3,0,5,1,2,7::2,3,0,6,1,5,7,4::2,6,7,0,1,3,5,4::7,6,4,0,2,3,5,1 --custom_align_max_joints=8

# train on bodies:  [501 506 513 500 511 505 503 502]

# ==> Alignment :  5,3,1,0,7,6,4,2::4,5,6,2,1,3,7,0::0,4,7,5,3,6,2,1::7,0,2,6,4,1,5,3::0,1,2,6,3,5,7,4::0,7,4,6,2,3,1,5::4,3,1,7,5,0,6,2::2,5,0,6,4,7,1,3
# 039e591c0a457c937367f018359becef
sbatch -J $exp_name $submit_to python 1.train.py --seed=4587 -f=$exp_name/8_randomized/ --train_bodies=501,506,513,500,511,505,503,502 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,3,1,0,7,6,4,2::4,5,6,2,1,3,7,0::0,4,7,5,3,6,2,1::7,0,2,6,4,1,5,3::0,1,2,6,3,5,7,4::0,7,4,6,2,3,1,5::4,3,1,7,5,0,6,2::2,5,0,6,4,7,1,3 --custom_align_max_joints=8

# train on bodies:  [503 501 507 511 502 515 504 500]

# ==> Alignment :  7,5,3,1,0,6,4,2::6,3,5,4,0,7,1,2::1,0,3,7,6,4,2,5::6,7,2,1,4,5,0,3::6,0,3,4,2,5,7,1::4,5,0,6,2,3,1,7::3,4,7,1,0,2,5,6::7,6,3,2,0,4,1,5
# 6351d94bb0c3fed70e16ff88dc8ea21d
sbatch -J $exp_name $submit_to python 1.train.py --seed=2527 -f=$exp_name/8_randomized/ --train_bodies=503,501,507,511,502,515,504,500 --topology_wrapper=CustomAlignWrapper --custom_alignment=7,5,3,1,0,6,4,2::6,3,5,4,0,7,1,2::1,0,3,7,6,4,2,5::6,7,2,1,4,5,0,3::6,0,3,4,2,5,7,1::4,5,0,6,2,3,1,7::3,4,7,1,0,2,5,6::7,6,3,2,0,4,1,5 --custom_align_max_joints=8

# train on bodies:  [505 510 504 514 509 503 508 507]

# ==> Alignment :  2,4,1,0,5,3,7,6::7,5,6,0,3,2,1,4::0,3,5,2,7,6,4,1::1,2,3,6,5,7,4,0::5,4,6,7,1,3,2,0::2,1,0,6,3,7,5,4::7,0,6,3,2,1,5,4::6,4,7,3,2,1,0,5
# 3a47173beb46704f239020b38d85ed38
sbatch -J $exp_name $submit_to python 1.train.py --seed=9181 -f=$exp_name/8_randomized/ --train_bodies=505,510,504,514,509,503,508,507 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,1,0,5,3,7,6::7,5,6,0,3,2,1,4::0,3,5,2,7,6,4,1::1,2,3,6,5,7,4,0::5,4,6,7,1,3,2,0::2,1,0,6,3,7,5,4::7,0,6,3,2,1,5,4::6,4,7,3,2,1,0,5 --custom_align_max_joints=8

# train on bodies:  [513 501 500 503 510 505 502 504]

# ==> Alignment :  2,5,7,3,0,1,4,6::3,6,5,7,0,2,4,1::4,0,2,7,5,6,3,1::6,1,3,7,2,4,5,0::3,4,0,6,1,2,5,7::7,3,6,0,5,2,4,1::5,1,2,6,3,4,0,7::5,7,2,1,3,4,6,0
# 606308a2be9fc5502dca3c18f55236fa
sbatch -J $exp_name $submit_to python 1.train.py --seed=5646 -f=$exp_name/8_randomized/ --train_bodies=513,501,500,503,510,505,502,504 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,5,7,3,0,1,4,6::3,6,5,7,0,2,4,1::4,0,2,7,5,6,3,1::6,1,3,7,2,4,5,0::3,4,0,6,1,2,5,7::7,3,6,0,5,2,4,1::5,1,2,6,3,4,0,7::5,7,2,1,3,4,6,0 --custom_align_max_joints=8

# train on bodies:  [515 502 506 505 501 500 508 503]

# ==> Alignment :  7,5,1,0,6,4,2,3::6,1,2,7,4,0,5,3::2,5,7,1,0,3,4,6::6,4,7,3,2,1,0,5::1,0,3,7,5,2,6,4::4,7,3,5,0,1,6,2::7,1,4,5,6,2,3,0::6,4,3,2,1,5,7,0
# 6e384e5d1bc24ab0847ab8e3ed91777e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8956 -f=$exp_name/8_randomized/ --train_bodies=515,502,506,505,501,500,508,503 --topology_wrapper=CustomAlignWrapper --custom_alignment=7,5,1,0,6,4,2,3::6,1,2,7,4,0,5,3::2,5,7,1,0,3,4,6::6,4,7,3,2,1,0,5::1,0,3,7,5,2,6,4::4,7,3,5,0,1,6,2::7,1,4,5,6,2,3,0::6,4,3,2,1,5,7,0 --custom_align_max_joints=8

# train on bodies:  [512 501 508 506 503 502 500 514]

# ==> Alignment :  5,6,0,2,4,3,7,1::3,6,1,7,2,0,5,4::6,2,0,1,4,5,3,7::3,7,5,6,0,4,1,2::2,0,3,5,4,7,1,6::6,1,5,4,3,2,0,7::3,1,7,0,6,5,2,4::2,5,1,4,0,7,6,3
# 9be74be1a6e85b161ffb2a2872401b4e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1018 -f=$exp_name/8_randomized/ --train_bodies=512,501,508,506,503,502,500,514 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,6,0,2,4,3,7,1::3,6,1,7,2,0,5,4::6,2,0,1,4,5,3,7::3,7,5,6,0,4,1,2::2,0,3,5,4,7,1,6::6,1,5,4,3,2,0,7::3,1,7,0,6,5,2,4::2,5,1,4,0,7,6,3 --custom_align_max_joints=8

# train on bodies:  [604 602 612 611 614 609 600 610]

# ==> Alignment :  0,2,1::1,2,0::0,2,1::0,2,1::1,2,0::1,2,0::1,0,2::0,2,1
# 07c2fb16b0c6cdfdf2d12c27a9630b39
sbatch -J $exp_name $submit_to python 1.train.py --seed=8787 -f=$exp_name/8_randomized/ --train_bodies=604,602,612,611,614,609,600,610 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::1,2,0::0,2,1::0,2,1::1,2,0::1,2,0::1,0,2::0,2,1 --custom_align_max_joints=3

# train on bodies:  [614 608 609 605 604 612 600 613]

# ==> Alignment :  0,2,1::2,0,1::2,1,0::2,0,1::2,0,1::0,1,2::1,2,0::2,0,1
# 5291ee16b6e8d07ee0eb52f5e21db15e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9256 -f=$exp_name/8_randomized/ --train_bodies=614,608,609,605,604,612,600,613 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::2,0,1::2,1,0::2,0,1::2,0,1::0,1,2::1,2,0::2,0,1 --custom_align_max_joints=3

# train on bodies:  [602 614 610 605 609 607 600 612]

# ==> Alignment :  1,2,0::1,0,2::2,1,0::2,1,0::1,0,2::2,1,0::2,0,1::1,2,0
# 1b319cf452da4ad045dfb06c8b37b193
sbatch -J $exp_name $submit_to python 1.train.py --seed=9704 -f=$exp_name/8_randomized/ --train_bodies=602,614,610,605,609,607,600,612 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,2,0::1,0,2::2,1,0::2,1,0::1,0,2::2,1,0::2,0,1::1,2,0 --custom_align_max_joints=3

# train on bodies:  [613 604 608 601 614 615 606 612]

# ==> Alignment :  1,2,0::1,0,2::1,0,2::0,1,2::2,1,0::1,2,0::1,2,0::1,0,2
# 3b96c344fefe14a7727e580257e27e6d
sbatch -J $exp_name $submit_to python 1.train.py --seed=6175 -f=$exp_name/8_randomized/ --train_bodies=613,604,608,601,614,615,606,612 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,2,0::1,0,2::1,0,2::0,1,2::2,1,0::1,2,0::1,2,0::1,0,2 --custom_align_max_joints=3

# train on bodies:  [608 610 611 615 607 605 606 614]

# ==> Alignment :  2,0,1::0,1,2::2,0,1::0,1,2::1,2,0::0,1,2::2,0,1::2,0,1
# 632c5c4353efe38058fd0c501c20afb8
sbatch -J $exp_name $submit_to python 1.train.py --seed=347 -f=$exp_name/8_randomized/ --train_bodies=608,610,611,615,607,605,606,614 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,1::0,1,2::2,0,1::0,1,2::1,2,0::0,1,2::2,0,1::2,0,1 --custom_align_max_joints=3

# train on bodies:  [601 602 605 608 607 610 613 611]

# ==> Alignment :  0,2,1::1,0,2::0,2,1::2,1,0::0,2,1::1,0,2::0,2,1::0,2,1
# 6505561d4c0266056ed4c28572d38d98
sbatch -J $exp_name $submit_to python 1.train.py --seed=39 -f=$exp_name/8_randomized/ --train_bodies=601,602,605,608,607,610,613,611 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::1,0,2::0,2,1::2,1,0::0,2,1::1,0,2::0,2,1::0,2,1 --custom_align_max_joints=3

# train on bodies:  [615 610 612 609 608 613 603 607]

# ==> Alignment :  2,1,0::2,0,1::1,0,2::0,1,2::1,0,2::2,1,0::1,2,0::0,1,2
# a501724db9a8ed3a81b2e0c1628de825
sbatch -J $exp_name $submit_to python 1.train.py --seed=3674 -f=$exp_name/8_randomized/ --train_bodies=615,610,612,609,608,613,603,607 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,1,0::2,0,1::1,0,2::0,1,2::1,0,2::2,1,0::1,2,0::0,1,2 --custom_align_max_joints=3

# train on bodies:  [613 612 608 607 615 600 610 609]

# ==> Alignment :  0,2,1::0,2,1::2,1,0::1,0,2::2,0,1::0,2,1::2,1,0::1,2,0
# c7c8e447179b530fcce46eda35737f2c
sbatch -J $exp_name $submit_to python 1.train.py --seed=8857 -f=$exp_name/8_randomized/ --train_bodies=613,612,608,607,615,600,610,609 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::0,2,1::2,1,0::1,0,2::2,0,1::0,2,1::2,1,0::1,2,0 --custom_align_max_joints=3

# train on bodies:  [607 600 605 611 614 615 609 613]

# ==> Alignment :  1,2,0::2,0,1::0,1,2::0,2,1::1,2,0::1,0,2::0,1,2::0,1,2
# 355c6406ad189ac5b40e5d5272d5b04a
sbatch -J $exp_name $submit_to python 1.train.py --seed=4019 -f=$exp_name/8_randomized/ --train_bodies=607,600,605,611,614,615,609,613 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,2,0::2,0,1::0,1,2::0,2,1::1,2,0::1,0,2::0,1,2::0,1,2 --custom_align_max_joints=3

# train on bodies:  [605 610 611 604 607 614 603 615]

# ==> Alignment :  1,2,0::1,0,2::0,2,1::2,1,0::1,2,0::2,1,0::2,0,1::2,0,1
# 90f6c9c5afb9b91ff7c530656e3a7502
sbatch -J $exp_name $submit_to python 1.train.py --seed=7174 -f=$exp_name/8_randomized/ --train_bodies=605,610,611,604,607,614,603,615 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,2,0::1,0,2::0,2,1::2,1,0::1,2,0::2,1,0::2,0,1::2,0,1 --custom_align_max_joints=3

# train on bodies:  [305 313 307 301 312 303 308 315 309 310 306 302 304 314 300 311]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=899 -f=$exp_name/16_aligned/ --train_bodies=305,313,307,301,312,303,308,315,309,310,306,302,304,314,300,311 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [315 311 303 307 313 305 308 314 312 310 306 309 302 301 300 304]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8535 -f=$exp_name/16_aligned/ --train_bodies=315,311,303,307,313,305,308,314,312,310,306,309,302,301,300,304 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [301 305 308 314 300 307 310 303 313 309 304 315 302 311 306 312]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3473 -f=$exp_name/16_aligned/ --train_bodies=301,305,308,314,300,307,310,303,313,309,304,315,302,311,306,312 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [308 302 300 301 315 311 303 312 304 305 313 309 310 306 307 314]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5249 -f=$exp_name/16_aligned/ --train_bodies=308,302,300,301,315,311,303,312,304,305,313,309,310,306,307,314 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [308 312 310 309 307 311 306 305 315 314 313 300 304 301 302 303]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6500 -f=$exp_name/16_aligned/ --train_bodies=308,312,310,309,307,311,306,305,315,314,313,300,304,301,302,303 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [315 309 306 314 303 304 307 313 312 305 301 300 308 310 311 302]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7375 -f=$exp_name/16_aligned/ --train_bodies=315,309,306,314,303,304,307,313,312,305,301,300,308,310,311,302 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [303 307 308 309 313 310 312 300 305 301 315 311 306 302 304 314]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7563 -f=$exp_name/16_aligned/ --train_bodies=303,307,308,309,313,310,312,300,305,301,315,311,306,302,304,314 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [312 313 304 310 303 309 315 308 301 311 302 314 300 306 307 305]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3963 -f=$exp_name/16_aligned/ --train_bodies=312,313,304,310,303,309,315,308,301,311,302,314,300,306,307,305 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [311 302 306 308 307 304 314 305 310 301 315 309 303 313 312 300]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5603 -f=$exp_name/16_aligned/ --train_bodies=311,302,306,308,307,304,314,305,310,301,315,309,303,313,312,300 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [308 313 309 307 314 304 302 305 312 306 310 315 300 303 311 301]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1907 -f=$exp_name/16_aligned/ --train_bodies=308,313,309,307,314,304,302,305,312,306,310,315,300,303,311,301 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [400 411 414 407 408 402 409 410 413 401 412 404 406 415 403 405]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8510 -f=$exp_name/16_aligned/ --train_bodies=400,411,414,407,408,402,409,410,413,401,412,404,406,415,403,405 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [400 407 408 406 409 410 404 415 405 412 403 402 401 411 414 413]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6618 -f=$exp_name/16_aligned/ --train_bodies=400,407,408,406,409,410,404,415,405,412,403,402,401,411,414,413 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [403 412 406 405 415 409 408 414 400 402 407 411 401 410 413 404]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2934 -f=$exp_name/16_aligned/ --train_bodies=403,412,406,405,415,409,408,414,400,402,407,411,401,410,413,404 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [413 401 404 405 407 414 403 409 402 411 415 410 406 412 408 400]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8447 -f=$exp_name/16_aligned/ --train_bodies=413,401,404,405,407,414,403,409,402,411,415,410,406,412,408,400 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [401 412 400 406 409 402 407 405 410 403 413 414 415 411 408 404]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1175 -f=$exp_name/16_aligned/ --train_bodies=401,412,400,406,409,402,407,405,410,403,413,414,415,411,408,404 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [410 407 409 403 402 411 406 405 414 400 415 413 412 408 401 404]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7084 -f=$exp_name/16_aligned/ --train_bodies=410,407,409,403,402,411,406,405,414,400,415,413,412,408,401,404 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [415 403 404 401 410 412 406 413 407 411 414 400 408 409 405 402]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4861 -f=$exp_name/16_aligned/ --train_bodies=415,403,404,401,410,412,406,413,407,411,414,400,408,409,405,402 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [401 412 402 408 409 403 415 406 400 407 410 413 414 404 411 405]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1160 -f=$exp_name/16_aligned/ --train_bodies=401,412,402,408,409,403,415,406,400,407,410,413,414,404,411,405 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [408 415 406 404 411 409 413 410 403 414 402 405 400 401 407 412]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7324 -f=$exp_name/16_aligned/ --train_bodies=408,415,406,404,411,409,413,410,403,414,402,405,400,401,407,412 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [410 413 404 407 414 403 411 406 408 400 401 405 402 409 415 412]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1881 -f=$exp_name/16_aligned/ --train_bodies=410,413,404,407,414,403,411,406,408,400,401,405,402,409,415,412 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=6

# train on bodies:  [500 502 515 506 514 509 510 503 501 507 504 505 508 512 513 511]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4917 -f=$exp_name/16_aligned/ --train_bodies=500,502,515,506,514,509,510,503,501,507,504,505,508,512,513,511 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [513 503 506 507 500 502 508 505 504 512 511 510 501 509 515 514]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5894 -f=$exp_name/16_aligned/ --train_bodies=513,503,506,507,500,502,508,505,504,512,511,510,501,509,515,514 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [510 503 502 511 512 504 514 505 500 513 501 506 507 509 508 515]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9648 -f=$exp_name/16_aligned/ --train_bodies=510,503,502,511,512,504,514,505,500,513,501,506,507,509,508,515 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [504 515 508 506 510 513 505 512 503 514 500 501 507 502 511 509]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9318 -f=$exp_name/16_aligned/ --train_bodies=504,515,508,506,510,513,505,512,503,514,500,501,507,502,511,509 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [511 515 510 512 514 507 503 509 505 508 500 506 513 501 504 502]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7891 -f=$exp_name/16_aligned/ --train_bodies=511,515,510,512,514,507,503,509,505,508,500,506,513,501,504,502 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [501 507 508 504 513 506 502 511 514 510 515 503 509 505 512 500]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7245 -f=$exp_name/16_aligned/ --train_bodies=501,507,508,504,513,506,502,511,514,510,515,503,509,505,512,500 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [509 504 508 501 513 500 503 515 506 512 514 507 511 510 505 502]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=6304 -f=$exp_name/16_aligned/ --train_bodies=509,504,508,501,513,500,503,515,506,512,514,507,511,510,505,502 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [513 500 502 510 511 509 512 508 515 501 507 514 503 504 506 505]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7620 -f=$exp_name/16_aligned/ --train_bodies=513,500,502,510,511,509,512,508,515,501,507,514,503,504,506,505 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [513 505 507 515 512 508 501 510 500 511 514 509 504 502 503 506]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=2989 -f=$exp_name/16_aligned/ --train_bodies=513,505,507,515,512,508,501,510,500,511,514,509,504,502,503,506 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [508 500 507 506 509 504 514 501 503 513 515 512 510 505 511 502]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1651 -f=$exp_name/16_aligned/ --train_bodies=508,500,507,506,509,504,514,501,503,513,515,512,510,505,511,502 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=8

# train on bodies:  [610 612 609 615 603 607 613 604 611 605 601 602 600 606 614 608]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=5139 -f=$exp_name/16_aligned/ --train_bodies=610,612,609,615,603,607,613,604,611,605,601,602,600,606,614,608 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [605 603 601 602 600 604 609 608 611 614 606 610 607 613 612 615]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3768 -f=$exp_name/16_aligned/ --train_bodies=605,603,601,602,600,604,609,608,611,614,606,610,607,613,612,615 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [612 607 613 608 604 611 602 603 605 601 614 606 609 600 615 610]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=3912 -f=$exp_name/16_aligned/ --train_bodies=612,607,613,608,604,611,602,603,605,601,614,606,609,600,615,610 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [600 603 608 615 610 612 602 613 604 611 601 605 606 607 614 609]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4848 -f=$exp_name/16_aligned/ --train_bodies=600,603,608,615,610,612,602,613,604,611,601,605,606,607,614,609 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [603 613 607 614 605 601 600 608 609 611 604 612 615 610 602 606]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=7192 -f=$exp_name/16_aligned/ --train_bodies=603,613,607,614,605,601,600,608,609,611,604,612,615,610,602,606 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [603 604 606 601 602 613 605 609 615 612 608 611 600 607 610 614]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=4237 -f=$exp_name/16_aligned/ --train_bodies=603,604,606,601,602,613,605,609,615,612,608,611,600,607,610,614 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [606 609 608 605 614 615 600 612 607 611 602 613 604 610 601 603]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=1026 -f=$exp_name/16_aligned/ --train_bodies=606,609,608,605,614,615,600,612,607,611,602,613,604,610,601,603 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [603 605 612 601 615 613 614 602 609 610 607 608 611 600 604 606]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8831 -f=$exp_name/16_aligned/ --train_bodies=603,605,612,601,615,613,614,602,609,610,607,608,611,600,604,606 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [606 609 603 614 604 612 607 602 601 611 608 615 613 610 605 600]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=8895 -f=$exp_name/16_aligned/ --train_bodies=606,609,603,614,604,612,607,602,601,611,608,615,613,610,605,600 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [611 603 613 615 609 614 612 605 601 610 606 604 600 607 608 602]

# ==> Alignment :  
# d41d8cd98f00b204e9800998ecf8427e
sbatch -J $exp_name $submit_to python 1.train.py --seed=9386 -f=$exp_name/16_aligned/ --train_bodies=611,603,613,615,609,614,612,605,601,610,606,604,600,607,608,602 --topology_wrapper=CustomAlignWrapper --custom_alignment= --custom_align_max_joints=3

# train on bodies:  [307 308 303 300 302 304 312 309 301 310 306 305 315 313 314 311]

# ==> Alignment :  2,4,0,1,5,3::3,1,2,0,5,4::3,2,5,0,1,4::0,4,2,1,3,5::2,5,0,1,4,3::0,2,1,5,4,3::4,2,3,0,1,5::1,3,2,4,5,0::3,2,4,5,1,0::5,1,3,4,2,0::1,0,4,5,2,3::5,3,1,4,0,2::3,5,2,0,1,4::2,0,4,3,5,1::1,3,0,4,5,2::0,5,1,4,2,3
# 9ad6a17820b4a2bf7eafa9142850597a
sbatch -J $exp_name $submit_to python 1.train.py --seed=9473 -f=$exp_name/16_randomized/ --train_bodies=307,308,303,300,302,304,312,309,301,310,306,305,315,313,314,311 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,0,1,5,3::3,1,2,0,5,4::3,2,5,0,1,4::0,4,2,1,3,5::2,5,0,1,4,3::0,2,1,5,4,3::4,2,3,0,1,5::1,3,2,4,5,0::3,2,4,5,1,0::5,1,3,4,2,0::1,0,4,5,2,3::5,3,1,4,0,2::3,5,2,0,1,4::2,0,4,3,5,1::1,3,0,4,5,2::0,5,1,4,2,3 --custom_align_max_joints=6

# train on bodies:  [305 304 310 301 303 314 307 306 302 312 300 313 309 308 315 311]

# ==> Alignment :  4,1,2,0,5,3::1,3,2,0,5,4::4,1,0,5,2,3::3,0,2,4,1,5::1,5,2,4,3,0::3,5,0,4,1,2::2,3,5,4,0,1::4,0,1,2,5,3::3,2,4,5,0,1::4,5,1,2,0,3::1,5,4,3,0,2::0,4,5,1,3,2::0,5,3,1,2,4::0,4,3,5,2,1::1,5,4,3,0,2::5,2,4,1,0,3
# 3cdc9482de43f2e0c222da9cee76e9ff
sbatch -J $exp_name $submit_to python 1.train.py --seed=4089 -f=$exp_name/16_randomized/ --train_bodies=305,304,310,301,303,314,307,306,302,312,300,313,309,308,315,311 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,1,2,0,5,3::1,3,2,0,5,4::4,1,0,5,2,3::3,0,2,4,1,5::1,5,2,4,3,0::3,5,0,4,1,2::2,3,5,4,0,1::4,0,1,2,5,3::3,2,4,5,0,1::4,5,1,2,0,3::1,5,4,3,0,2::0,4,5,1,3,2::0,5,3,1,2,4::0,4,3,5,2,1::1,5,4,3,0,2::5,2,4,1,0,3 --custom_align_max_joints=6

# train on bodies:  [311 301 315 314 304 309 302 313 307 312 305 310 306 308 303 300]

# ==> Alignment :  1,4,2,5,3,0::5,4,1,3,2,0::1,4,5,0,3,2::1,3,0,2,5,4::0,5,2,1,3,4::5,0,1,3,2,4::1,2,5,0,3,4::4,0,5,2,3,1::0,1,3,5,4,2::4,1,5,2,3,0::2,3,1,0,4,5::1,4,2,0,5,3::4,3,1,2,5,0::2,4,5,3,0,1::2,1,4,3,5,0::3,5,0,4,1,2
# 1fe2cab5de14f3e853fa3311f2577aa1
sbatch -J $exp_name $submit_to python 1.train.py --seed=9733 -f=$exp_name/16_randomized/ --train_bodies=311,301,315,314,304,309,302,313,307,312,305,310,306,308,303,300 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,4,2,5,3,0::5,4,1,3,2,0::1,4,5,0,3,2::1,3,0,2,5,4::0,5,2,1,3,4::5,0,1,3,2,4::1,2,5,0,3,4::4,0,5,2,3,1::0,1,3,5,4,2::4,1,5,2,3,0::2,3,1,0,4,5::1,4,2,0,5,3::4,3,1,2,5,0::2,4,5,3,0,1::2,1,4,3,5,0::3,5,0,4,1,2 --custom_align_max_joints=6

# train on bodies:  [305 303 306 314 302 309 301 307 311 304 308 300 315 313 310 312]

# ==> Alignment :  5,0,3,1,2,4::3,0,1,4,5,2::5,4,1,2,0,3::0,3,2,1,4,5::2,4,5,0,1,3::2,0,5,3,4,1::2,5,3,0,1,4::5,1,2,0,4,3::2,3,0,1,4,5::2,1,0,5,4,3::0,5,2,1,3,4::5,0,1,4,2,3::2,0,4,3,5,1::3,4,0,5,2,1::2,1,5,4,3,0::5,1,2,4,0,3
# db9a102ed83efd45cd06b5cb7de09dd8
sbatch -J $exp_name $submit_to python 1.train.py --seed=6132 -f=$exp_name/16_randomized/ --train_bodies=305,303,306,314,302,309,301,307,311,304,308,300,315,313,310,312 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,0,3,1,2,4::3,0,1,4,5,2::5,4,1,2,0,3::0,3,2,1,4,5::2,4,5,0,1,3::2,0,5,3,4,1::2,5,3,0,1,4::5,1,2,0,4,3::2,3,0,1,4,5::2,1,0,5,4,3::0,5,2,1,3,4::5,0,1,4,2,3::2,0,4,3,5,1::3,4,0,5,2,1::2,1,5,4,3,0::5,1,2,4,0,3 --custom_align_max_joints=6

# train on bodies:  [315 310 312 302 311 306 314 313 304 309 307 303 308 300 301 305]

# ==> Alignment :  2,4,3,0,1,5::3,1,5,2,4,0::3,2,5,4,0,1::2,5,0,4,1,3::3,1,0,5,4,2::0,5,1,4,2,3::3,1,4,5,2,0::2,1,4,5,3,0::1,0,2,3,5,4::0,2,1,3,4,5::2,0,3,4,1,5::4,5,0,3,1,2::5,3,2,1,0,4::5,4,3,2,1,0::4,2,0,1,5,3::0,3,4,1,2,5
# e80688e2cf52a00d135539a176d4a093
sbatch -J $exp_name $submit_to python 1.train.py --seed=7200 -f=$exp_name/16_randomized/ --train_bodies=315,310,312,302,311,306,314,313,304,309,307,303,308,300,301,305 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,3,0,1,5::3,1,5,2,4,0::3,2,5,4,0,1::2,5,0,4,1,3::3,1,0,5,4,2::0,5,1,4,2,3::3,1,4,5,2,0::2,1,4,5,3,0::1,0,2,3,5,4::0,2,1,3,4,5::2,0,3,4,1,5::4,5,0,3,1,2::5,3,2,1,0,4::5,4,3,2,1,0::4,2,0,1,5,3::0,3,4,1,2,5 --custom_align_max_joints=6

# train on bodies:  [309 314 304 308 301 315 306 307 303 302 300 310 313 305 312 311]

# ==> Alignment :  2,0,5,1,4,3::4,0,5,3,1,2::5,2,1,3,0,4::0,5,1,4,3,2::5,3,1,2,4,0::3,1,5,4,2,0::5,0,1,2,3,4::2,1,0,4,3,5::3,0,4,5,2,1::3,1,2,4,0,5::1,2,4,0,5,3::4,3,2,1,5,0::2,5,0,3,4,1::3,5,1,4,0,2::5,4,1,2,3,0::0,1,2,3,5,4
# 80feca9f98d8cafff8b1e2412287d621
sbatch -J $exp_name $submit_to python 1.train.py --seed=7427 -f=$exp_name/16_randomized/ --train_bodies=309,314,304,308,301,315,306,307,303,302,300,310,313,305,312,311 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,5,1,4,3::4,0,5,3,1,2::5,2,1,3,0,4::0,5,1,4,3,2::5,3,1,2,4,0::3,1,5,4,2,0::5,0,1,2,3,4::2,1,0,4,3,5::3,0,4,5,2,1::3,1,2,4,0,5::1,2,4,0,5,3::4,3,2,1,5,0::2,5,0,3,4,1::3,5,1,4,0,2::5,4,1,2,3,0::0,1,2,3,5,4 --custom_align_max_joints=6

# train on bodies:  [310 307 309 305 302 308 304 301 300 303 312 315 314 311 313 306]

# ==> Alignment :  1,0,2,4,3,5::5,2,3,0,1,4::1,2,0,5,3,4::4,2,3,0,5,1::1,3,5,4,0,2::0,1,2,3,5,4::0,2,4,3,5,1::3,5,1,0,2,4::1,3,5,0,2,4::1,4,0,2,5,3::5,3,1,4,0,2::4,3,5,1,2,0::1,5,3,2,0,4::1,2,5,3,4,0::3,1,0,5,4,2::4,0,3,5,2,1
# cf0d365bd1d00aaa3ff5543ad2690ee0
sbatch -J $exp_name $submit_to python 1.train.py --seed=87 -f=$exp_name/16_randomized/ --train_bodies=310,307,309,305,302,308,304,301,300,303,312,315,314,311,313,306 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,0,2,4,3,5::5,2,3,0,1,4::1,2,0,5,3,4::4,2,3,0,5,1::1,3,5,4,0,2::0,1,2,3,5,4::0,2,4,3,5,1::3,5,1,0,2,4::1,3,5,0,2,4::1,4,0,2,5,3::5,3,1,4,0,2::4,3,5,1,2,0::1,5,3,2,0,4::1,2,5,3,4,0::3,1,0,5,4,2::4,0,3,5,2,1 --custom_align_max_joints=6

# train on bodies:  [308 306 311 301 313 300 310 302 314 315 307 303 312 304 305 309]

# ==> Alignment :  3,2,5,0,4,1::0,4,2,3,1,5::0,2,3,4,1,5::5,2,4,0,3,1::0,4,2,5,1,3::4,1,5,2,0,3::0,1,3,2,4,5::0,4,3,5,1,2::1,2,5,0,3,4::4,5,2,3,0,1::0,3,2,4,5,1::5,0,1,2,3,4::2,5,4,3,0,1::5,0,3,2,4,1::5,0,4,2,1,3::5,2,0,3,4,1
# 49b41125c9cd18d4ad52b57da7f47992
sbatch -J $exp_name $submit_to python 1.train.py --seed=4236 -f=$exp_name/16_randomized/ --train_bodies=308,306,311,301,313,300,310,302,314,315,307,303,312,304,305,309 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,2,5,0,4,1::0,4,2,3,1,5::0,2,3,4,1,5::5,2,4,0,3,1::0,4,2,5,1,3::4,1,5,2,0,3::0,1,3,2,4,5::0,4,3,5,1,2::1,2,5,0,3,4::4,5,2,3,0,1::0,3,2,4,5,1::5,0,1,2,3,4::2,5,4,3,0,1::5,0,3,2,4,1::5,0,4,2,1,3::5,2,0,3,4,1 --custom_align_max_joints=6

# train on bodies:  [302 300 301 315 311 305 312 303 314 308 309 310 304 307 313 306]

# ==> Alignment :  0,3,2,1,4,5::4,2,3,5,1,0::4,1,0,3,2,5::2,5,0,3,1,4::3,0,1,5,4,2::5,0,3,1,4,2::3,4,1,0,5,2::4,0,3,1,5,2::4,2,1,5,3,0::3,2,1,5,4,0::0,2,4,5,3,1::4,3,2,1,0,5::4,2,0,5,1,3::4,5,1,0,3,2::1,5,4,2,0,3::2,4,1,5,3,0
# 12e4d23df0bc9bbc16f6062466907f34
sbatch -J $exp_name $submit_to python 1.train.py --seed=4549 -f=$exp_name/16_randomized/ --train_bodies=302,300,301,315,311,305,312,303,314,308,309,310,304,307,313,306 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,3,2,1,4,5::4,2,3,5,1,0::4,1,0,3,2,5::2,5,0,3,1,4::3,0,1,5,4,2::5,0,3,1,4,2::3,4,1,0,5,2::4,0,3,1,5,2::4,2,1,5,3,0::3,2,1,5,4,0::0,2,4,5,3,1::4,3,2,1,0,5::4,2,0,5,1,3::4,5,1,0,3,2::1,5,4,2,0,3::2,4,1,5,3,0 --custom_align_max_joints=6

# train on bodies:  [302 304 306 313 312 314 311 309 301 305 303 315 310 300 308 307]

# ==> Alignment :  4,1,3,2,5,0::4,3,0,5,1,2::0,2,1,3,4,5::0,5,4,3,1,2::0,4,1,3,2,5::3,4,0,1,2,5::5,0,4,1,2,3::5,1,4,2,0,3::4,3,0,5,1,2::2,3,0,5,1,4::5,0,3,4,2,1::0,4,5,2,1,3::2,3,0,1,4,5::5,2,3,1,4,0::2,3,0,5,1,4::5,1,3,0,2,4
# 5bf68df1faf123dd20f2e4b1c041e3ba
sbatch -J $exp_name $submit_to python 1.train.py --seed=7944 -f=$exp_name/16_randomized/ --train_bodies=302,304,306,313,312,314,311,309,301,305,303,315,310,300,308,307 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,1,3,2,5,0::4,3,0,5,1,2::0,2,1,3,4,5::0,5,4,3,1,2::0,4,1,3,2,5::3,4,0,1,2,5::5,0,4,1,2,3::5,1,4,2,0,3::4,3,0,5,1,2::2,3,0,5,1,4::5,0,3,4,2,1::0,4,5,2,1,3::2,3,0,1,4,5::5,2,3,1,4,0::2,3,0,5,1,4::5,1,3,0,2,4 --custom_align_max_joints=6

# train on bodies:  [401 411 414 413 410 406 412 407 405 409 404 400 415 403 402 408]

# ==> Alignment :  4,5,1,3,2,0::0,1,3,5,2,4::1,0,3,5,2,4::5,3,2,4,0,1::3,1,5,2,0,4::4,2,0,5,3,1::5,2,4,0,1,3::3,5,0,1,4,2::4,2,1,0,5,3::4,5,0,3,2,1::5,2,3,0,4,1::0,5,4,1,3,2::0,1,5,4,3,2::4,5,3,0,1,2::1,0,5,4,3,2::3,1,5,0,2,4
# 708ba3d2370edac1c48962b3b364cc8d
sbatch -J $exp_name $submit_to python 1.train.py --seed=8149 -f=$exp_name/16_randomized/ --train_bodies=401,411,414,413,410,406,412,407,405,409,404,400,415,403,402,408 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,5,1,3,2,0::0,1,3,5,2,4::1,0,3,5,2,4::5,3,2,4,0,1::3,1,5,2,0,4::4,2,0,5,3,1::5,2,4,0,1,3::3,5,0,1,4,2::4,2,1,0,5,3::4,5,0,3,2,1::5,2,3,0,4,1::0,5,4,1,3,2::0,1,5,4,3,2::4,5,3,0,1,2::1,0,5,4,3,2::3,1,5,0,2,4 --custom_align_max_joints=6

# train on bodies:  [412 400 415 406 405 409 401 411 404 407 414 413 410 403 408 402]

# ==> Alignment :  4,1,5,3,0,2::5,1,2,3,0,4::2,5,4,0,3,1::0,2,5,1,4,3::1,2,4,5,0,3::0,4,5,1,2,3::3,0,1,4,2,5::0,3,2,4,5,1::5,4,1,2,3,0::0,2,4,1,5,3::5,4,0,2,1,3::0,1,4,3,2,5::4,0,1,2,5,3::5,1,3,2,0,4::3,4,0,2,1,5::2,3,5,1,4,0
# da2a76291c6a3f3dd1ac9a9bce5f5e00
sbatch -J $exp_name $submit_to python 1.train.py --seed=6380 -f=$exp_name/16_randomized/ --train_bodies=412,400,415,406,405,409,401,411,404,407,414,413,410,403,408,402 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,1,5,3,0,2::5,1,2,3,0,4::2,5,4,0,3,1::0,2,5,1,4,3::1,2,4,5,0,3::0,4,5,1,2,3::3,0,1,4,2,5::0,3,2,4,5,1::5,4,1,2,3,0::0,2,4,1,5,3::5,4,0,2,1,3::0,1,4,3,2,5::4,0,1,2,5,3::5,1,3,2,0,4::3,4,0,2,1,5::2,3,5,1,4,0 --custom_align_max_joints=6

# train on bodies:  [402 408 400 411 401 412 413 403 410 407 404 409 415 406 405 414]

# ==> Alignment :  2,4,3,0,1,5::3,1,2,0,4,5::3,2,0,5,1,4::4,1,0,2,5,3::2,4,0,3,1,5::5,4,1,2,3,0::1,4,5,2,0,3::2,4,0,5,1,3::2,3,5,1,0,4::4,0,2,1,5,3::0,4,3,5,2,1::2,0,4,5,3,1::5,3,0,2,4,1::2,4,5,0,3,1::3,0,4,2,1,5::5,3,2,1,4,0
# 9fc806bb842a53322abef6ed7047a34e
sbatch -J $exp_name $submit_to python 1.train.py --seed=808 -f=$exp_name/16_randomized/ --train_bodies=402,408,400,411,401,412,413,403,410,407,404,409,415,406,405,414 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,4,3,0,1,5::3,1,2,0,4,5::3,2,0,5,1,4::4,1,0,2,5,3::2,4,0,3,1,5::5,4,1,2,3,0::1,4,5,2,0,3::2,4,0,5,1,3::2,3,5,1,0,4::4,0,2,1,5,3::0,4,3,5,2,1::2,0,4,5,3,1::5,3,0,2,4,1::2,4,5,0,3,1::3,0,4,2,1,5::5,3,2,1,4,0 --custom_align_max_joints=6

# train on bodies:  [403 413 402 401 407 411 415 408 400 406 404 412 405 410 409 414]

# ==> Alignment :  4,0,3,5,2,1::5,0,3,2,4,1::2,4,3,0,5,1::5,2,4,0,3,1::3,0,4,2,1,5::5,4,2,3,0,1::5,3,1,4,2,0::3,5,0,4,1,2::2,3,5,1,0,4::2,1,3,5,0,4::3,2,1,4,5,0::1,2,4,3,0,5::4,2,0,1,5,3::2,3,4,0,1,5::2,0,3,1,4,5::3,4,0,2,5,1
# e76d1524519d7c3aeffec5ee54858676
sbatch -J $exp_name $submit_to python 1.train.py --seed=864 -f=$exp_name/16_randomized/ --train_bodies=403,413,402,401,407,411,415,408,400,406,404,412,405,410,409,414 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,0,3,5,2,1::5,0,3,2,4,1::2,4,3,0,5,1::5,2,4,0,3,1::3,0,4,2,1,5::5,4,2,3,0,1::5,3,1,4,2,0::3,5,0,4,1,2::2,3,5,1,0,4::2,1,3,5,0,4::3,2,1,4,5,0::1,2,4,3,0,5::4,2,0,1,5,3::2,3,4,0,1,5::2,0,3,1,4,5::3,4,0,2,5,1 --custom_align_max_joints=6

# train on bodies:  [407 405 409 400 414 403 411 408 410 401 415 413 412 406 402 404]

# ==> Alignment :  1,3,0,5,2,4::4,0,5,1,3,2::3,2,0,4,5,1::2,1,4,3,0,5::1,0,3,2,4,5::5,3,4,2,0,1::1,5,2,4,3,0::3,0,5,4,1,2::1,4,5,0,2,3::2,1,4,5,0,3::0,5,4,1,3,2::5,0,4,3,1,2::2,1,3,4,0,5::5,1,4,0,3,2::5,2,0,1,3,4::0,1,5,2,4,3
# 558be857b87fedbf7e1203101ed995c1
sbatch -J $exp_name $submit_to python 1.train.py --seed=9585 -f=$exp_name/16_randomized/ --train_bodies=407,405,409,400,414,403,411,408,410,401,415,413,412,406,402,404 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,3,0,5,2,4::4,0,5,1,3,2::3,2,0,4,5,1::2,1,4,3,0,5::1,0,3,2,4,5::5,3,4,2,0,1::1,5,2,4,3,0::3,0,5,4,1,2::1,4,5,0,2,3::2,1,4,5,0,3::0,5,4,1,3,2::5,0,4,3,1,2::2,1,3,4,0,5::5,1,4,0,3,2::5,2,0,1,3,4::0,1,5,2,4,3 --custom_align_max_joints=6

# train on bodies:  [410 402 415 405 408 407 401 411 400 406 412 413 403 409 414 404]

# ==> Alignment :  0,2,5,3,1,4::1,4,2,5,0,3::5,3,4,2,0,1::0,2,3,4,1,5::0,5,3,2,1,4::2,5,1,4,3,0::4,2,0,3,5,1::3,1,2,4,5,0::4,3,5,2,1,0::5,2,1,3,0,4::1,5,0,2,4,3::3,0,1,5,4,2::3,0,5,2,4,1::0,2,4,1,5,3::0,1,4,3,5,2::0,4,3,1,5,2
# e64d4166b32101db8ff02d66e54f2943
sbatch -J $exp_name $submit_to python 1.train.py --seed=920 -f=$exp_name/16_randomized/ --train_bodies=410,402,415,405,408,407,401,411,400,406,412,413,403,409,414,404 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,5,3,1,4::1,4,2,5,0,3::5,3,4,2,0,1::0,2,3,4,1,5::0,5,3,2,1,4::2,5,1,4,3,0::4,2,0,3,5,1::3,1,2,4,5,0::4,3,5,2,1,0::5,2,1,3,0,4::1,5,0,2,4,3::3,0,1,5,4,2::3,0,5,2,4,1::0,2,4,1,5,3::0,1,4,3,5,2::0,4,3,1,5,2 --custom_align_max_joints=6

# train on bodies:  [408 401 415 406 414 412 402 404 413 410 407 400 409 411 405 403]

# ==> Alignment :  2,0,4,1,5,3::0,3,4,5,2,1::0,5,3,1,4,2::2,5,3,0,1,4::2,1,3,0,4,5::5,2,1,0,3,4::4,5,2,1,0,3::4,0,5,3,2,1::0,1,5,4,2,3::1,0,2,3,4,5::4,5,1,2,0,3::3,1,0,5,4,2::5,1,2,0,3,4::1,3,4,5,0,2::1,5,3,2,4,0::1,4,2,3,5,0
# 7a1fc128937df867d382ebf337f47ce7
sbatch -J $exp_name $submit_to python 1.train.py --seed=2771 -f=$exp_name/16_randomized/ --train_bodies=408,401,415,406,414,412,402,404,413,410,407,400,409,411,405,403 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,4,1,5,3::0,3,4,5,2,1::0,5,3,1,4,2::2,5,3,0,1,4::2,1,3,0,4,5::5,2,1,0,3,4::4,5,2,1,0,3::4,0,5,3,2,1::0,1,5,4,2,3::1,0,2,3,4,5::4,5,1,2,0,3::3,1,0,5,4,2::5,1,2,0,3,4::1,3,4,5,0,2::1,5,3,2,4,0::1,4,2,3,5,0 --custom_align_max_joints=6

# train on bodies:  [400 410 401 402 412 414 405 413 403 408 411 415 404 406 407 409]

# ==> Alignment :  2,0,3,5,4,1::4,1,3,2,5,0::4,2,1,3,5,0::4,5,2,1,0,3::5,0,1,4,2,3::2,1,5,4,3,0::5,0,4,3,2,1::1,4,0,3,5,2::3,5,2,4,0,1::5,3,4,2,0,1::4,3,0,2,5,1::2,4,3,0,1,5::1,2,4,0,5,3::2,5,0,4,1,3::3,2,0,5,1,4::3,2,4,1,5,0
# d271a4b3d84cb3cc2e6a574cf42aed73
sbatch -J $exp_name $submit_to python 1.train.py --seed=3425 -f=$exp_name/16_randomized/ --train_bodies=400,410,401,402,412,414,405,413,403,408,411,415,404,406,407,409 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,3,5,4,1::4,1,3,2,5,0::4,2,1,3,5,0::4,5,2,1,0,3::5,0,1,4,2,3::2,1,5,4,3,0::5,0,4,3,2,1::1,4,0,3,5,2::3,5,2,4,0,1::5,3,4,2,0,1::4,3,0,2,5,1::2,4,3,0,1,5::1,2,4,0,5,3::2,5,0,4,1,3::3,2,0,5,1,4::3,2,4,1,5,0 --custom_align_max_joints=6

# train on bodies:  [401 409 404 405 411 414 408 403 400 402 410 407 406 413 415 412]

# ==> Alignment :  0,3,2,1,5,4::3,0,4,1,2,5::5,0,2,4,3,1::0,2,3,1,5,4::0,2,1,4,5,3::4,0,5,1,2,3::4,5,1,2,3,0::4,0,2,1,5,3::5,4,2,1,0,3::2,0,3,4,1,5::3,2,1,0,4,5::4,5,2,3,1,0::3,4,2,0,5,1::2,5,4,3,0,1::2,0,1,5,4,3::2,4,3,0,1,5
# 745bfbb69c53ac597a0395e18838e75b
sbatch -J $exp_name $submit_to python 1.train.py --seed=6238 -f=$exp_name/16_randomized/ --train_bodies=401,409,404,405,411,414,408,403,400,402,410,407,406,413,415,412 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,3,2,1,5,4::3,0,4,1,2,5::5,0,2,4,3,1::0,2,3,1,5,4::0,2,1,4,5,3::4,0,5,1,2,3::4,5,1,2,3,0::4,0,2,1,5,3::5,4,2,1,0,3::2,0,3,4,1,5::3,2,1,0,4,5::4,5,2,3,1,0::3,4,2,0,5,1::2,5,4,3,0,1::2,0,1,5,4,3::2,4,3,0,1,5 --custom_align_max_joints=6

# train on bodies:  [405 409 406 413 415 403 401 404 408 402 400 411 407 410 414 412]

# ==> Alignment :  1,5,0,3,2,4::3,0,4,5,2,1::2,1,3,4,5,0::1,0,3,5,2,4::0,5,3,1,2,4::5,0,4,1,3,2::3,0,2,4,1,5::0,4,1,3,5,2::3,2,5,0,1,4::4,3,0,1,2,5::0,2,1,5,4,3::3,4,1,2,5,0::1,5,2,0,4,3::0,5,3,4,2,1::1,5,0,2,3,4::2,1,4,3,0,5
# 869913a73c180b09870cfca1625b108b
sbatch -J $exp_name $submit_to python 1.train.py --seed=241 -f=$exp_name/16_randomized/ --train_bodies=405,409,406,413,415,403,401,404,408,402,400,411,407,410,414,412 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,5,0,3,2,4::3,0,4,5,2,1::2,1,3,4,5,0::1,0,3,5,2,4::0,5,3,1,2,4::5,0,4,1,3,2::3,0,2,4,1,5::0,4,1,3,5,2::3,2,5,0,1,4::4,3,0,1,2,5::0,2,1,5,4,3::3,4,1,2,5,0::1,5,2,0,4,3::0,5,3,4,2,1::1,5,0,2,3,4::2,1,4,3,0,5 --custom_align_max_joints=6

# train on bodies:  [508 511 504 506 501 513 514 515 509 500 510 512 502 503 507 505]

# ==> Alignment :  6,1,0,4,7,2,3,5::5,0,7,1,4,3,2,6::1,4,7,0,5,3,6,2::4,5,3,0,2,1,7,6::6,7,2,3,1,5,4,0::1,3,2,4,5,6,7,0::6,3,4,2,1,7,5,0::2,5,3,6,0,1,4,7::6,5,7,4,0,2,3,1::4,1,3,5,2,7,0,6::2,5,7,3,4,0,6,1::6,5,4,0,3,7,2,1::1,0,5,4,6,7,3,2::3,7,0,1,4,5,2,6::1,6,5,3,2,4,0,7::7,0,6,2,1,4,5,3
# f2c87d721987288b9af1afe78bd0ef38
sbatch -J $exp_name $submit_to python 1.train.py --seed=5925 -f=$exp_name/16_randomized/ --train_bodies=508,511,504,506,501,513,514,515,509,500,510,512,502,503,507,505 --topology_wrapper=CustomAlignWrapper --custom_alignment=6,1,0,4,7,2,3,5::5,0,7,1,4,3,2,6::1,4,7,0,5,3,6,2::4,5,3,0,2,1,7,6::6,7,2,3,1,5,4,0::1,3,2,4,5,6,7,0::6,3,4,2,1,7,5,0::2,5,3,6,0,1,4,7::6,5,7,4,0,2,3,1::4,1,3,5,2,7,0,6::2,5,7,3,4,0,6,1::6,5,4,0,3,7,2,1::1,0,5,4,6,7,3,2::3,7,0,1,4,5,2,6::1,6,5,3,2,4,0,7::7,0,6,2,1,4,5,3 --custom_align_max_joints=8

# train on bodies:  [505 504 510 515 511 501 500 503 506 508 509 502 507 512 514 513]

# ==> Alignment :  3,7,0,1,2,4,6,5::4,2,6,5,0,7,3,1::4,2,3,0,7,5,1,6::6,4,1,0,5,7,2,3::7,0,3,1,4,6,2,5::7,0,2,4,6,1,3,5::3,4,0,2,1,6,7,5::2,4,5,0,6,3,1,7::3,5,6,7,4,2,1,0::6,5,3,7,1,4,0,2::6,2,5,1,4,0,3,7::3,2,5,0,7,4,1,6::4,2,5,7,6,3,1,0::0,4,2,6,5,7,1,3::7,4,0,3,1,6,5,2::4,5,7,3,6,0,2,1
# 021ae2e43ac773355837e3b2da3710d7
sbatch -J $exp_name $submit_to python 1.train.py --seed=4517 -f=$exp_name/16_randomized/ --train_bodies=505,504,510,515,511,501,500,503,506,508,509,502,507,512,514,513 --topology_wrapper=CustomAlignWrapper --custom_alignment=3,7,0,1,2,4,6,5::4,2,6,5,0,7,3,1::4,2,3,0,7,5,1,6::6,4,1,0,5,7,2,3::7,0,3,1,4,6,2,5::7,0,2,4,6,1,3,5::3,4,0,2,1,6,7,5::2,4,5,0,6,3,1,7::3,5,6,7,4,2,1,0::6,5,3,7,1,4,0,2::6,2,5,1,4,0,3,7::3,2,5,0,7,4,1,6::4,2,5,7,6,3,1,0::0,4,2,6,5,7,1,3::7,4,0,3,1,6,5,2::4,5,7,3,6,0,2,1 --custom_align_max_joints=8

# train on bodies:  [503 500 506 504 515 502 511 505 509 512 514 508 513 501 507 510]

# ==> Alignment :  4,7,5,3,0,6,1,2::0,4,3,5,1,6,2,7::1,4,7,3,2,0,6,5::5,1,6,4,2,3,7,0::6,5,3,1,0,7,4,2::2,4,6,0,5,7,1,3::7,5,2,6,0,3,1,4::6,7,4,2,1,0,3,5::2,4,7,6,0,5,3,1::0,6,5,7,2,1,4,3::4,1,5,2,0,6,3,7::6,1,5,4,0,2,3,7::6,3,2,4,5,1,7,0::3,5,2,0,6,4,1,7::7,3,2,5,1,6,0,4::1,5,3,0,2,6,7,4
# 44fdaeeb884e1402ebac8b1759362b06
sbatch -J $exp_name $submit_to python 1.train.py --seed=500 -f=$exp_name/16_randomized/ --train_bodies=503,500,506,504,515,502,511,505,509,512,514,508,513,501,507,510 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,7,5,3,0,6,1,2::0,4,3,5,1,6,2,7::1,4,7,3,2,0,6,5::5,1,6,4,2,3,7,0::6,5,3,1,0,7,4,2::2,4,6,0,5,7,1,3::7,5,2,6,0,3,1,4::6,7,4,2,1,0,3,5::2,4,7,6,0,5,3,1::0,6,5,7,2,1,4,3::4,1,5,2,0,6,3,7::6,1,5,4,0,2,3,7::6,3,2,4,5,1,7,0::3,5,2,0,6,4,1,7::7,3,2,5,1,6,0,4::1,5,3,0,2,6,7,4 --custom_align_max_joints=8

# train on bodies:  [513 501 512 507 509 500 515 504 508 506 511 503 510 505 514 502]

# ==> Alignment :  0,1,4,3,7,5,6,2::1,5,2,4,7,0,3,6::7,4,1,6,0,5,3,2::4,2,5,1,3,6,0,7::6,4,3,2,0,7,1,5::5,4,7,6,3,2,0,1::6,5,2,1,0,7,4,3::2,4,0,7,6,5,1,3::3,0,1,2,4,5,6,7::4,2,1,7,5,3,0,6::1,0,7,2,3,4,6,5::0,1,5,3,6,4,7,2::4,0,3,2,7,1,5,6::6,0,4,7,1,2,5,3::6,1,7,0,3,2,4,5::4,6,2,5,3,7,0,1
# 0d124518fb48e8ceb68c771b4626ebdc
sbatch -J $exp_name $submit_to python 1.train.py --seed=919 -f=$exp_name/16_randomized/ --train_bodies=513,501,512,507,509,500,515,504,508,506,511,503,510,505,514,502 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,1,4,3,7,5,6,2::1,5,2,4,7,0,3,6::7,4,1,6,0,5,3,2::4,2,5,1,3,6,0,7::6,4,3,2,0,7,1,5::5,4,7,6,3,2,0,1::6,5,2,1,0,7,4,3::2,4,0,7,6,5,1,3::3,0,1,2,4,5,6,7::4,2,1,7,5,3,0,6::1,0,7,2,3,4,6,5::0,1,5,3,6,4,7,2::4,0,3,2,7,1,5,6::6,0,4,7,1,2,5,3::6,1,7,0,3,2,4,5::4,6,2,5,3,7,0,1 --custom_align_max_joints=8

# train on bodies:  [503 514 515 504 506 513 501 512 509 507 510 502 505 511 500 508]

# ==> Alignment :  1,5,4,6,2,3,7,0::2,1,6,7,4,5,3,0::1,0,3,4,5,6,7,2::0,6,5,7,3,4,2,1::3,6,2,5,4,0,1,7::5,4,0,1,6,7,3,2::3,0,7,6,2,4,1,5::0,4,3,1,5,6,7,2::6,3,2,7,4,0,1,5::1,0,2,7,4,5,6,3::1,3,6,4,7,5,2,0::1,3,7,4,6,5,0,2::4,0,6,1,2,5,3,7::5,2,1,3,7,4,6,0::2,4,7,0,5,6,1,3::2,1,4,0,7,3,6,5
# e22c11170ab21fec6d1c210cf7f830a3
sbatch -J $exp_name $submit_to python 1.train.py --seed=4859 -f=$exp_name/16_randomized/ --train_bodies=503,514,515,504,506,513,501,512,509,507,510,502,505,511,500,508 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,5,4,6,2,3,7,0::2,1,6,7,4,5,3,0::1,0,3,4,5,6,7,2::0,6,5,7,3,4,2,1::3,6,2,5,4,0,1,7::5,4,0,1,6,7,3,2::3,0,7,6,2,4,1,5::0,4,3,1,5,6,7,2::6,3,2,7,4,0,1,5::1,0,2,7,4,5,6,3::1,3,6,4,7,5,2,0::1,3,7,4,6,5,0,2::4,0,6,1,2,5,3,7::5,2,1,3,7,4,6,0::2,4,7,0,5,6,1,3::2,1,4,0,7,3,6,5 --custom_align_max_joints=8

# train on bodies:  [504 507 500 505 511 503 501 514 515 508 509 512 502 510 506 513]

# ==> Alignment :  0,3,1,4,7,2,6,5::7,4,3,5,2,0,1,6::2,4,0,7,5,3,1,6::0,4,2,6,1,5,3,7::0,6,2,4,3,1,7,5::4,2,1,0,6,7,3,5::6,3,0,7,1,2,4,5::4,2,0,1,7,3,6,5::4,1,6,0,7,5,2,3::0,5,2,1,4,6,3,7::5,0,3,1,2,4,7,6::3,7,1,4,5,0,6,2::6,5,7,4,1,3,2,0::5,4,0,7,3,2,6,1::2,5,0,3,1,4,7,6::1,5,4,7,6,2,0,3
# 59a21b4d296dd8b1ad50deb0a023106d
sbatch -J $exp_name $submit_to python 1.train.py --seed=5265 -f=$exp_name/16_randomized/ --train_bodies=504,507,500,505,511,503,501,514,515,508,509,512,502,510,506,513 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,3,1,4,7,2,6,5::7,4,3,5,2,0,1,6::2,4,0,7,5,3,1,6::0,4,2,6,1,5,3,7::0,6,2,4,3,1,7,5::4,2,1,0,6,7,3,5::6,3,0,7,1,2,4,5::4,2,0,1,7,3,6,5::4,1,6,0,7,5,2,3::0,5,2,1,4,6,3,7::5,0,3,1,2,4,7,6::3,7,1,4,5,0,6,2::6,5,7,4,1,3,2,0::5,4,0,7,3,2,6,1::2,5,0,3,1,4,7,6::1,5,4,7,6,2,0,3 --custom_align_max_joints=8

# train on bodies:  [515 502 506 504 503 512 514 505 509 511 500 507 510 508 513 501]

# ==> Alignment :  4,7,6,3,2,0,5,1::7,6,2,3,4,5,0,1::4,2,0,3,6,5,7,1::1,5,0,6,7,3,4,2::3,7,6,4,1,2,5,0::7,4,2,1,3,0,6,5::4,0,2,3,1,6,7,5::5,4,3,2,0,6,1,7::5,4,1,0,7,2,6,3::7,4,6,0,1,3,5,2::2,1,5,7,0,4,6,3::5,2,6,3,0,4,1,7::4,5,3,6,1,7,2,0::3,7,5,6,1,0,4,2::3,4,6,1,5,0,2,7::7,5,4,6,3,1,2,0
# 565da8ee5acf6aa6e4c9311bfa706f35
sbatch -J $exp_name $submit_to python 1.train.py --seed=9347 -f=$exp_name/16_randomized/ --train_bodies=515,502,506,504,503,512,514,505,509,511,500,507,510,508,513,501 --topology_wrapper=CustomAlignWrapper --custom_alignment=4,7,6,3,2,0,5,1::7,6,2,3,4,5,0,1::4,2,0,3,6,5,7,1::1,5,0,6,7,3,4,2::3,7,6,4,1,2,5,0::7,4,2,1,3,0,6,5::4,0,2,3,1,6,7,5::5,4,3,2,0,6,1,7::5,4,1,0,7,2,6,3::7,4,6,0,1,3,5,2::2,1,5,7,0,4,6,3::5,2,6,3,0,4,1,7::4,5,3,6,1,7,2,0::3,7,5,6,1,0,4,2::3,4,6,1,5,0,2,7::7,5,4,6,3,1,2,0 --custom_align_max_joints=8

# train on bodies:  [507 514 501 500 502 506 504 510 512 515 513 508 509 503 511 505]

# ==> Alignment :  2,7,4,0,1,6,3,5::5,4,1,2,0,6,7,3::7,5,3,6,4,2,0,1::2,3,4,5,7,1,0,6::3,0,4,6,5,1,2,7::1,6,0,5,2,7,4,3::0,1,3,2,6,5,7,4::3,6,4,1,5,7,0,2::3,7,1,0,2,4,5,6::2,3,5,7,4,1,6,0::5,0,6,1,2,7,3,4::5,4,2,1,6,3,7,0::7,1,6,0,5,2,3,4::5,7,6,0,4,3,2,1::7,6,0,3,4,2,5,1::7,1,4,2,5,6,0,3
# 47a1229e33ed38612cf7a7f080638b27
sbatch -J $exp_name $submit_to python 1.train.py --seed=263 -f=$exp_name/16_randomized/ --train_bodies=507,514,501,500,502,506,504,510,512,515,513,508,509,503,511,505 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,7,4,0,1,6,3,5::5,4,1,2,0,6,7,3::7,5,3,6,4,2,0,1::2,3,4,5,7,1,0,6::3,0,4,6,5,1,2,7::1,6,0,5,2,7,4,3::0,1,3,2,6,5,7,4::3,6,4,1,5,7,0,2::3,7,1,0,2,4,5,6::2,3,5,7,4,1,6,0::5,0,6,1,2,7,3,4::5,4,2,1,6,3,7,0::7,1,6,0,5,2,3,4::5,7,6,0,4,3,2,1::7,6,0,3,4,2,5,1::7,1,4,2,5,6,0,3 --custom_align_max_joints=8

# train on bodies:  [504 508 503 513 506 500 509 510 514 502 512 505 511 501 515 507]

# ==> Alignment :  0,5,6,2,4,3,1,7::4,7,2,5,1,0,6,3::5,1,3,6,2,4,0,7::5,3,6,2,0,4,7,1::2,1,0,6,5,3,7,4::4,2,3,7,6,1,5,0::7,3,1,6,0,5,2,4::7,3,1,2,0,6,5,4::2,5,3,1,6,7,4,0::4,1,3,7,0,6,5,2::2,3,4,5,1,6,0,7::1,3,4,5,2,0,7,6::2,5,1,3,7,0,6,4::3,7,5,0,6,2,1,4::4,5,0,2,6,1,3,7::0,5,7,2,1,4,3,6
# 6e484c0051e76cc76213469ab4a995eb
sbatch -J $exp_name $submit_to python 1.train.py --seed=3905 -f=$exp_name/16_randomized/ --train_bodies=504,508,503,513,506,500,509,510,514,502,512,505,511,501,515,507 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,5,6,2,4,3,1,7::4,7,2,5,1,0,6,3::5,1,3,6,2,4,0,7::5,3,6,2,0,4,7,1::2,1,0,6,5,3,7,4::4,2,3,7,6,1,5,0::7,3,1,6,0,5,2,4::7,3,1,2,0,6,5,4::2,5,3,1,6,7,4,0::4,1,3,7,0,6,5,2::2,3,4,5,1,6,0,7::1,3,4,5,2,0,7,6::2,5,1,3,7,0,6,4::3,7,5,0,6,2,1,4::4,5,0,2,6,1,3,7::0,5,7,2,1,4,3,6 --custom_align_max_joints=8

# train on bodies:  [501 513 505 503 509 510 507 515 504 512 502 514 511 506 508 500]

# ==> Alignment :  5,1,4,2,6,3,7,0::2,1,7,4,3,5,6,0::7,1,3,4,2,5,0,6::0,6,7,1,5,3,2,4::4,3,2,6,1,7,0,5::3,6,4,1,2,5,7,0::3,0,1,2,6,4,7,5::4,5,7,3,1,6,0,2::6,1,4,3,2,7,5,0::7,5,4,2,1,6,3,0::2,3,5,7,1,6,0,4::1,7,2,6,5,4,3,0::4,0,6,1,5,3,7,2::7,1,4,6,5,0,2,3::6,7,5,2,1,4,0,3::0,7,5,1,3,6,4,2
# e9bef0ebfa7744f300926722f1e31972
sbatch -J $exp_name $submit_to python 1.train.py --seed=4785 -f=$exp_name/16_randomized/ --train_bodies=501,513,505,503,509,510,507,515,504,512,502,514,511,506,508,500 --topology_wrapper=CustomAlignWrapper --custom_alignment=5,1,4,2,6,3,7,0::2,1,7,4,3,5,6,0::7,1,3,4,2,5,0,6::0,6,7,1,5,3,2,4::4,3,2,6,1,7,0,5::3,6,4,1,2,5,7,0::3,0,1,2,6,4,7,5::4,5,7,3,1,6,0,2::6,1,4,3,2,7,5,0::7,5,4,2,1,6,3,0::2,3,5,7,1,6,0,4::1,7,2,6,5,4,3,0::4,0,6,1,5,3,7,2::7,1,4,6,5,0,2,3::6,7,5,2,1,4,0,3::0,7,5,1,3,6,4,2 --custom_align_max_joints=8

# train on bodies:  [608 610 611 604 603 601 606 607 614 612 609 602 600 605 613 615]

# ==> Alignment :  0,2,1::2,0,1::0,2,1::1,0,2::1,2,0::1,2,0::0,1,2::1,0,2::0,2,1::0,1,2::0,1,2::1,2,0::2,1,0::2,1,0::0,2,1::0,1,2
# c0c90fad28d4d6e6063f8150107fd785
sbatch -J $exp_name $submit_to python 1.train.py --seed=4818 -f=$exp_name/16_randomized/ --train_bodies=608,610,611,604,603,601,606,607,614,612,609,602,600,605,613,615 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::2,0,1::0,2,1::1,0,2::1,2,0::1,2,0::0,1,2::1,0,2::0,2,1::0,1,2::0,1,2::1,2,0::2,1,0::2,1,0::0,2,1::0,1,2 --custom_align_max_joints=3

# train on bodies:  [601 608 604 603 611 600 609 615 612 614 607 602 613 606 610 605]

# ==> Alignment :  2,0,1::0,2,1::0,2,1::1,0,2::1,2,0::2,0,1::0,2,1::0,2,1::0,1,2::1,0,2::1,0,2::2,0,1::2,1,0::2,1,0::1,2,0::0,1,2
# 07399e4874c043d73619de029d2cec3c
sbatch -J $exp_name $submit_to python 1.train.py --seed=3735 -f=$exp_name/16_randomized/ --train_bodies=601,608,604,603,611,600,609,615,612,614,607,602,613,606,610,605 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,1::0,2,1::0,2,1::1,0,2::1,2,0::2,0,1::0,2,1::0,2,1::0,1,2::1,0,2::1,0,2::2,0,1::2,1,0::2,1,0::1,2,0::0,1,2 --custom_align_max_joints=3

# train on bodies:  [610 604 606 605 608 615 607 609 614 612 600 603 611 602 613 601]

# ==> Alignment :  0,2,1::0,2,1::1,2,0::1,2,0::1,2,0::1,2,0::1,0,2::2,0,1::2,1,0::0,1,2::2,1,0::0,1,2::2,1,0::2,0,1::1,0,2::1,0,2
# 4d95f10abbbf7ffe412dec2007a96172
sbatch -J $exp_name $submit_to python 1.train.py --seed=826 -f=$exp_name/16_randomized/ --train_bodies=610,604,606,605,608,615,607,609,614,612,600,603,611,602,613,601 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::0,2,1::1,2,0::1,2,0::1,2,0::1,2,0::1,0,2::2,0,1::2,1,0::0,1,2::2,1,0::0,1,2::2,1,0::2,0,1::1,0,2::1,0,2 --custom_align_max_joints=3

# train on bodies:  [602 613 615 612 608 614 610 611 607 603 600 604 606 605 609 601]

# ==> Alignment :  0,2,1::1,0,2::0,2,1::2,1,0::1,0,2::2,1,0::1,0,2::1,0,2::0,2,1::1,0,2::1,0,2::0,2,1::2,1,0::0,1,2::1,0,2::0,2,1
# d9dff76c557fb55879321498884a01a8
sbatch -J $exp_name $submit_to python 1.train.py --seed=28 -f=$exp_name/16_randomized/ --train_bodies=602,613,615,612,608,614,610,611,607,603,600,604,606,605,609,601 --topology_wrapper=CustomAlignWrapper --custom_alignment=0,2,1::1,0,2::0,2,1::2,1,0::1,0,2::2,1,0::1,0,2::1,0,2::0,2,1::1,0,2::1,0,2::0,2,1::2,1,0::0,1,2::1,0,2::0,2,1 --custom_align_max_joints=3

# train on bodies:  [606 605 613 610 604 602 611 609 614 603 612 600 615 607 608 601]

# ==> Alignment :  2,0,1::1,2,0::1,0,2::0,2,1::1,2,0::0,2,1::2,0,1::2,1,0::2,0,1::0,2,1::0,1,2::1,2,0::2,1,0::2,0,1::0,1,2::2,1,0
# 2a6f0058a871da34bfcb61b6effbf3a3
sbatch -J $exp_name $submit_to python 1.train.py --seed=5372 -f=$exp_name/16_randomized/ --train_bodies=606,605,613,610,604,602,611,609,614,603,612,600,615,607,608,601 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,0,1::1,2,0::1,0,2::0,2,1::1,2,0::0,2,1::2,0,1::2,1,0::2,0,1::0,2,1::0,1,2::1,2,0::2,1,0::2,0,1::0,1,2::2,1,0 --custom_align_max_joints=3

# train on bodies:  [604 606 601 600 611 609 610 602 614 615 613 603 607 608 605 612]

# ==> Alignment :  2,1,0::0,1,2::1,0,2::2,0,1::2,0,1::1,2,0::0,1,2::2,1,0::2,0,1::1,2,0::1,0,2::0,1,2::2,1,0::2,0,1::0,2,1::1,0,2
# afa0c9590e0df35d1ffed4ab558c7d10
sbatch -J $exp_name $submit_to python 1.train.py --seed=4829 -f=$exp_name/16_randomized/ --train_bodies=604,606,601,600,611,609,610,602,614,615,613,603,607,608,605,612 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,1,0::0,1,2::1,0,2::2,0,1::2,0,1::1,2,0::0,1,2::2,1,0::2,0,1::1,2,0::1,0,2::0,1,2::2,1,0::2,0,1::0,2,1::1,0,2 --custom_align_max_joints=3

# train on bodies:  [604 611 601 603 600 614 606 615 610 602 605 609 613 607 608 612]

# ==> Alignment :  1,2,0::1,0,2::1,0,2::1,0,2::2,0,1::2,1,0::2,0,1::2,1,0::2,1,0::1,2,0::0,1,2::2,1,0::0,1,2::2,0,1::2,0,1::0,1,2
# 29036d9f5bf8121ff7785e1440b9f055
sbatch -J $exp_name $submit_to python 1.train.py --seed=604 -f=$exp_name/16_randomized/ --train_bodies=604,611,601,603,600,614,606,615,610,602,605,609,613,607,608,612 --topology_wrapper=CustomAlignWrapper --custom_alignment=1,2,0::1,0,2::1,0,2::1,0,2::2,0,1::2,1,0::2,0,1::2,1,0::2,1,0::1,2,0::0,1,2::2,1,0::0,1,2::2,0,1::2,0,1::0,1,2 --custom_align_max_joints=3

# train on bodies:  [610 611 600 604 615 606 603 608 614 613 607 601 602 609 605 612]

# ==> Alignment :  2,1,0::0,2,1::0,1,2::0,2,1::0,1,2::1,0,2::2,0,1::1,0,2::0,1,2::1,0,2::2,1,0::1,2,0::0,1,2::0,2,1::0,2,1::2,1,0
# 76a6c95ac4d6dc0aee72a14a6ec16deb
sbatch -J $exp_name $submit_to python 1.train.py --seed=2862 -f=$exp_name/16_randomized/ --train_bodies=610,611,600,604,615,606,603,608,614,613,607,601,602,609,605,612 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,1,0::0,2,1::0,1,2::0,2,1::0,1,2::1,0,2::2,0,1::1,0,2::0,1,2::1,0,2::2,1,0::1,2,0::0,1,2::0,2,1::0,2,1::2,1,0 --custom_align_max_joints=3

# train on bodies:  [606 602 607 610 614 601 600 611 603 612 604 609 613 605 608 615]

# ==> Alignment :  2,1,0::2,0,1::1,0,2::1,2,0::1,0,2::2,0,1::2,1,0::1,2,0::0,1,2::0,1,2::1,0,2::2,0,1::1,0,2::2,0,1::1,0,2::0,1,2
# fdd83b484b6d1f607846e9fdc805f701
sbatch -J $exp_name $submit_to python 1.train.py --seed=338 -f=$exp_name/16_randomized/ --train_bodies=606,602,607,610,614,601,600,611,603,612,604,609,613,605,608,615 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,1,0::2,0,1::1,0,2::1,2,0::1,0,2::2,0,1::2,1,0::1,2,0::0,1,2::0,1,2::1,0,2::2,0,1::1,0,2::2,0,1::1,0,2::0,1,2 --custom_align_max_joints=3

# train on bodies:  [610 605 603 607 608 602 604 615 613 614 601 611 606 612 609 600]

# ==> Alignment :  2,1,0::0,1,2::2,0,1::2,1,0::1,2,0::2,0,1::0,2,1::2,1,0::2,1,0::0,2,1::1,0,2::0,2,1::2,1,0::1,0,2::0,1,2::1,0,2
# 7c5c034b96de3d9a7740504b5d86ab9a
sbatch -J $exp_name $submit_to python 1.train.py --seed=2706 -f=$exp_name/16_randomized/ --train_bodies=610,605,603,607,608,602,604,615,613,614,601,611,606,612,609,600 --topology_wrapper=CustomAlignWrapper --custom_alignment=2,1,0::0,1,2::2,0,1::2,1,0::1,2,0::2,0,1::0,2,1::2,1,0::2,1,0::0,2,1::1,0,2::0,2,1::2,1,0::1,0,2::0,1,2::1,0,2 --custom_align_max_joints=3

# In total: 320 jobs will be submitted.

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


4.