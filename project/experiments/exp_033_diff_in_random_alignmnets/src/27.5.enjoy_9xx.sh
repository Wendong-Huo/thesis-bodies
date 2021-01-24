#!/bin/sh

# good (m0! the assumed most meaningful alignment)
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --model_filename output_data/models/best_m0.zip --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --render
python 3.save_images_for_video.py --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --model_filename output_data/models/best_m0.zip --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --render --one_snapshot_at=40 --test_steps=50
montage output_data/saved_images/best_m0/test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 output_data/saved_images/best_m0/best_m0.png

# not that good
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --model_filename output_data/models/not_that_good.zip --custom_alignment=0,1,2,3,4,5,6,7::2,0,1,3,4,5,6,7::3,0,1,2,4,5,6,7::5,0,1,2,3,4,6,7::6,0,1,2,3,4,5,7::1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::2,3,0,1,4,5,6,7 --render

# mutant 2
python 2.test.py --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8 --custom_alignment=0,1,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,3,4,5,1,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --model_filename=output_data/tmp/1d8e72563daf848f1836d1effe995201-sd3180.zip



# best in mutate 4
python 2.test.py --model_filename output_data/models/best_in_m4.zip --custom_alignment=1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --custom_align_max_joints=8 --render --test_bodies=900,901,902,903,904,905,906,907  --topology_wrapper=CustomAlignWrapper
python 3.save_images_for_video.py --model_filename output_data/models/best_in_m4.zip --custom_alignment=1,0,2,3,4,5,6,7::1,2,0,3,4,5,6,7::1,2,3,0,4,5,6,7::1,2,3,0,5,4,6,7::7,2,3,4,5,6,0,1::1,7,2,3,4,5,6,0::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --custom_align_max_joints=8 --render --one_snapshot_at=40 --test_steps=50 --test_bodies=900,901,902,903,904,905,906,907  --topology_wrapper=CustomAlignWrapper
montage output_data/saved_images/best_in_m4/test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 output_data/saved_images/best_in_m4/best_in_m4.png
montage output_data/saved_images/best_in_m4/best_in_m4.png -geometry +0+0 -pointsize 80 -annotate +10+80 'best in m4' output_data/saved_images/best_in_m4/best_in_m4.png

# worst in mutate 4
python 2.test.py --model_filename output_data/models/worst_in_m4.zip --custom_alignment=0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7 --custom_align_max_joints=8 --render --test_bodies=900,901,902,903,904,905,906,907  --topology_wrapper=CustomAlignWrapper
python 3.save_images_for_video.py --model_filename output_data/models/worst_in_m4.zip --custom_alignment=0,7,2,3,4,5,6,1::1,7,0,3,4,5,6,2::1,2,3,0,4,5,6,7::3,2,1,4,5,0,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,6,7::2,0,1,3,4,5,6,7::2,3,0,1,5,4,6,7 --custom_align_max_joints=8 --render --one_snapshot_at=40 --test_steps=50 --test_bodies=900,901,902,903,904,905,906,907  --topology_wrapper=CustomAlignWrapper
montage output_data/saved_images/worst_in_m4/test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 output_data/saved_images/worst_in_m4/worst_in_m4.png
montage output_data/saved_images/worst_in_m4/worst_in_m4.png -geometry +0+0 -pointsize 80 -annotate +10+80 'worst in m4' output_data/saved_images/worst_in_m4/worst_in_m4.png


# best in mutate 8
python 2.test.py --model_filename output_data/models/best_in_m8.zip --custom_alignment=4,1,3,2,5,0,7,6::1,2,0,3,4,5,6,7::1,2,3,5,4,0,6,7::1,2,0,4,5,3,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --custom_align_max_joints=8 --render  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
python 3.save_images_for_video.py --model_filename output_data/models/best_in_m8.zip --custom_alignment=4,1,3,2,5,0,7,6::1,2,0,3,4,5,6,7::1,2,3,5,4,0,6,7::1,2,0,4,5,3,6,7::1,2,3,4,5,6,0,7::1,0,2,3,4,5,7,6::2,0,1,3,4,5,6,7::2,3,0,1,4,5,6,7 --custom_align_max_joints=8 --render --one_snapshot_at=40 --test_steps=50  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
montage output_data/saved_images/best_in_m8/test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 output_data/saved_images/best_in_m8/best_in_m8.png
montage output_data/saved_images/best_in_m8/best_in_m8.png -geometry +0+0 -pointsize 80 -annotate +10+80 'best in m8' output_data/saved_images/best_in_m8/best_in_m8.png

# worst in mutate 8
python 2.test.py --model_filename output_data/models/worst_in_m8.zip --custom_alignment=2,1,0,3,4,5,6,7::6,5,0,3,4,2,1,7::1,2,3,0,4,5,6,7::1,2,3,5,4,0,6,7::5,3,2,4,1,6,0,7::5,0,2,3,4,1,6,7::2,0,1,3,4,5,6,7::3,2,0,1,4,5,6,7 --custom_align_max_joints=8 --render  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
python 3.save_images_for_video.py --model_filename output_data/models/worst_in_m8.zip --custom_alignment=2,1,0,3,4,5,6,7::6,5,0,3,4,2,1,7::1,2,3,0,4,5,6,7::1,2,3,5,4,0,6,7::5,3,2,4,1,6,0,7::5,0,2,3,4,1,6,7::2,0,1,3,4,5,6,7::3,2,0,1,4,5,6,7 --custom_align_max_joints=8 --render --one_snapshot_at=40 --test_steps=50  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
montage output_data/saved_images/worst_in_m8/test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 output_data/saved_images/worst_in_m8/worst_in_m8.png
montage output_data/saved_images/worst_in_m8/worst_in_m8.png -geometry +0+0 -pointsize 80 -annotate +10+80 'worst in m8' output_data/saved_images/worst_in_m8/worst_in_m8.png



# best in mutate 16
# python 2.test.py --model_filename output_data/models/best_in_m16.zip --custom_alignment=0,1,2,3,4,5,6,7::6,1,0,2,4,5,3,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,0,4,3,1,5,7::1,6,2,3,4,5,0,7::2,0,1,6,4,7,5,3::2,3,0,1,4,5,6,7 --custom_align_max_joints=8 --render  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
python 3.save_images_for_video.py --model_filename output_data/models/best_in_m16.zip --custom_alignment=0,1,2,3,4,5,6,7::6,1,0,2,4,5,3,7::1,2,3,0,4,5,6,7::1,2,3,4,5,0,6,7::6,2,0,4,3,1,5,7::1,6,2,3,4,5,0,7::2,0,1,6,4,7,5,3::2,3,0,1,4,5,6,7 --custom_align_max_joints=8 --render --one_snapshot_at=40 --test_steps=50  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
montage output_data/saved_images/best_in_m16/test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 output_data/saved_images/best_in_m16/best_in_m16.png
montage output_data/saved_images/best_in_m16/best_in_m16.png -geometry +0+0 -pointsize 80 -annotate +10+80 'best in m16' output_data/saved_images/best_in_m16/best_in_m16.png

# worst in mutate 16
# python 2.test.py --model_filename output_data/models/worst_in_m16.zip --custom_alignment=7,1,2,3,4,5,6,0::6,2,1,3,4,0,5,7::1,0,3,2,4,5,6,7::1,3,6,2,4,0,5,7::1,2,3,4,7,6,5,0::4,3,2,5,1,0,6,7::5,0,6,3,4,2,1,7::2,3,0,1,4,5,6,7 --custom_align_max_joints=8 --render  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
python 3.save_images_for_video.py --model_filename output_data/models/worst_in_m16.zip --custom_alignment=7,1,2,3,4,5,6,0::6,2,1,3,4,0,5,7::1,0,3,2,4,5,6,7::1,3,6,2,4,0,5,7::1,2,3,4,7,6,5,0::4,3,2,5,1,0,6,7::5,0,6,3,4,2,1,7::2,3,0,1,4,5,6,7 --custom_align_max_joints=8 --render --one_snapshot_at=40 --test_steps=50  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
montage output_data/saved_images/worst_in_m16/test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 output_data/saved_images/worst_in_m16/worst_in_m16.png
montage output_data/saved_images/worst_in_m16/worst_in_m16.png -geometry +0+0 -pointsize 80 -annotate +10+80 'worst in m16' output_data/saved_images/worst_in_m16/worst_in_m16.png



# best in mutate 32
# python 2.test.py --model_filename output_data/models/best_in_m32.zip --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,4,7,5,6,3::3,6,1,5,4,0,2,7::1,6,3,5,4,7,2,0::6,2,0,4,5,1,3,7::0,7,2,5,3,4,1,6::2,0,7,3,4,5,1,6::2,3,0,4,6,5,1,7 --custom_align_max_joints=8 --render  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
python 3.save_images_for_video.py --model_filename output_data/models/best_in_m32.zip --custom_alignment=7,1,2,3,4,5,6,0::1,2,0,4,7,5,6,3::3,6,1,5,4,0,2,7::1,6,3,5,4,7,2,0::6,2,0,4,5,1,3,7::0,7,2,5,3,4,1,6::2,0,7,3,4,5,1,6::2,3,0,4,6,5,1,7 --custom_align_max_joints=8 --render --one_snapshot_at=40 --test_steps=50  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
montage output_data/saved_images/best_in_m32/test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 output_data/saved_images/best_in_m32/best_in_m32.png
montage output_data/saved_images/best_in_m32/best_in_m32.png -geometry +0+0 -pointsize 80 -annotate +10+80 'best in m32' output_data/saved_images/best_in_m32/best_in_m32.png

# worst in mutate 32
# python 2.test.py --model_filename output_data/models/worst_in_m32.zip --custom_alignment=0,7,6,5,3,2,4,1::5,2,3,4,1,0,6,7::3,1,2,0,4,6,5,7::7,4,1,2,0,5,6,3::7,2,3,5,0,6,4,1::1,4,2,6,3,5,0,7::4,0,1,3,2,5,6,7::2,4,0,1,3,5,6,7 --custom_align_max_joints=8 --render  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
python 3.save_images_for_video.py --model_filename output_data/models/worst_in_m32.zip --custom_alignment=0,7,6,5,3,2,4,1::5,2,3,4,1,0,6,7::3,1,2,0,4,6,5,7::7,4,1,2,0,5,6,3::7,2,3,5,0,6,4,1::1,4,2,6,3,5,0,7::4,0,1,3,2,5,6,7::2,4,0,1,3,5,6,7 --custom_align_max_joints=8 --render --one_snapshot_at=40 --test_steps=50  --test_bodies=900,901,902,903,904,905,906,907 --topology_wrapper=CustomAlignWrapper 
montage output_data/saved_images/worst_in_m32/test_90*_00040.png -crop '600x600+660+240' -mode Concatenate -tile 4x2 output_data/saved_images/worst_in_m32/worst_in_m32.png
montage output_data/saved_images/worst_in_m32/worst_in_m32.png -geometry +0+0 -pointsize 80 -annotate +10+80 'worst in m32' output_data/saved_images/worst_in_m32/worst_in_m32.png
