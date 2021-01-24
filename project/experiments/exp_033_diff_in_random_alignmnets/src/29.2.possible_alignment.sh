alignment=1,5,0,4,6,7,3,2::3,7,0,6,2,4,5,1::5,4,7,0,1,6,2,3::3,7,5,4,0,6,2,1
python 1.train.py --train_steps=5e6 --seed=23199 --train_bodies=399,499,599,699 --test_bodies=399,499,599,699 --topology_wrapper=CustomAlignWrapper \
--custom_alignment=$alignment --custom_align_max_joints=8