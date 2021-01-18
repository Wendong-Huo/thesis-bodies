#!/bin/sh

# good
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --model_filename output_data/models/good.zip --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --render

# not that good
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --model_filename output_data/models/not_that_good.zip --custom_alignment=0,1,2,3,4,5,6,7::2,0,1,3,4,5,6,7::3,0,1,2,4,5,6,7::5,0,1,2,3,4,6,7::6,0,1,2,3,4,5,7::1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::2,3,0,1,4,5,6,7 --render

# mutant 2
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --model_filename=output_data/tmp/1d8e72563daf848f1836d1effe995201-sd3180.zip
