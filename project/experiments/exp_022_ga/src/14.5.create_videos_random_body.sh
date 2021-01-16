#!/bin/sh

set -x

#generate videos
for body in $(seq 100 107)
do
    ffmpeg -framerate 80 -i output_data/saved_images/model-$(echo $body)-sd0/test_$(echo $body)_00%3d.png -y output_data/videos/1xx_$body.mp4
    # ffmpeg -framerate 100 -i output_data/saved_images/9xx-not-aligned/test_$(echo $body)_00%3d.png -y output_data/videos/9nn_$body.mp4

    # ffmpeg -i output_data/videos/9xx_$body.mp4 -filter:v "crop=in_w/2:in_h:in_w/4:0" -c:a -y output_data/videos/1xx_$body.mp4
    # ffmpeg -i output_data/videos/9nn_$body.mp4 -filter:v "crop=in_w/2:in_h:in_w/4:0" -c:a -y output_data/videos/9nn_half_$body.mp4

    # ffmpeg -i output_data/videos/9xx_half_$body.mp4 -i output_data/videos/9nn_half_$body.mp4 -filter_complex \
    # "[0:v][1:v]hstack=shortest=1[step1]; \
    # [step1]drawtext=text='$body Meaningfully Aligned':fontfile=FreeSerif.ttf:fontcolor=black:fontsize=50:x=10:y=10[step2]; \
    # [step2]drawtext=text='$body Arbitrarily Aligned':fontfile=FreeSerif.ttf:fontcolor=black:fontsize=50:x=1920/2+10:y=10 \
    # " -y output_data/videos/compare_$body.mp4

done

ffmpeg \
-i output_data/videos/1xx_100.mp4 \
-i output_data/videos/1xx_101.mp4 \
-i output_data/videos/1xx_102.mp4 \
-i output_data/videos/1xx_103.mp4 \
-i output_data/videos/1xx_104.mp4 \
-i output_data/videos/1xx_105.mp4 \
-i output_data/videos/1xx_106.mp4 \
-i output_data/videos/1xx_107.mp4 \
 -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v][6:v][7:v] concat=n=8" -y output_data/videos/1xx_final.mp4


# ffmpeg -framerate 100 -i output_data/saved_images/barchart_00%3d.png -y output_data/videos/barchart_$body.mp4
# ffmpeg -i output_data/videos/getCameraImage_$body.mp4 -vf "movie=output_data/videos/barchart_$body.mp4, scale=500: -1 [inner];[in][inner]overlay=10:10 [out]" -y output_data/videos/$body.mp4

# tile into one video
# ffmpeg \
# 	-i output_data/videos/test_900.mp4 \
#     -i output_data/videos/test_901.mp4 \
#     -i output_data/videos/test_902.mp4 \
#     -i output_data/videos/test_903.mp4 \
# 	-filter_complex "nullsrc=size=1920x1080 [base];[0:v] setpts=PTS-STARTPTS, scale=960x540 [upperleft];[1:v] setpts=PTS-STARTPTS, scale=960x540 [upperright];[2:v] setpts=PTS-STARTPTS, scale=960x540 [lowerleft];[3:v] setpts=PTS-STARTPTS, scale=960x540 [lowerright];[base][upperleft] overlay=shortest=1 [tmp1];[tmp1][upperright] overlay=shortest=1:x=960 [tmp2];[tmp2][lowerleft] overlay=shortest=1:y=540 [tmp3];[tmp3][lowerright] overlay=shortest=1:x=960:y=540" \
# 	-y output_data/videos/tiled.mp4
