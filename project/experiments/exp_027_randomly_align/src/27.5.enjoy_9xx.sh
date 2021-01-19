#!/bin/sh

# good
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --model_filename output_data/models/good.zip --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --render

# not that good
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --model_filename output_data/models/not_that_good.zip --custom_alignment=0,1,2,3,4,5,6,7::2,0,1,3,4,5,6,7::3,0,1,2,4,5,6,7::5,0,1,2,3,4,6,7::6,0,1,2,3,4,5,7::1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::2,3,0,1,4,5,6,7 --render

# mutant 2
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --model_filename=output_data/tmp/1d8e72563daf848f1836d1effe995201-sd3180.zip



# best in mutate 4
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --model_filename output_data/models/best_in_m4.zip --topology_wrapper=CustomAlignWrapper --custom_alignment=1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --custom_align_max_joints=8 --render
python 3.save_images_for_video.py --test_bodies=900,901,902,903,904,905,906,907 --model_filename output_data/models/best_in_m4.zip --topology_wrapper=CustomAlignWrapper --custom_alignment=1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --custom_align_max_joints=8 --render --one_snapshot_at=40 --test_steps=50
montage test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 best_in_m4.png

# worst in mutate 4
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --model_filename output_data/models/worst_in_m4.zip --topology_wrapper=CustomAlignWrapper --custom_alignment=0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7 --custom_align_max_joints=8 --render
python 3.save_images_for_video.py --test_bodies=900,901,902,903,904,905,906,907 --model_filename output_data/models/worst_in_m4.zip --topology_wrapper=CustomAlignWrapper --custom_alignment=0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7 --custom_align_max_joints=8 --render --one_snapshot_at=40 --test_steps=50
montage test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 worst_in_m4.png