#!/bin/sh

set -x

#generate videos
cat output_data/saved_images/test_*.png | ffmpeg -framerate 100 -f image2pipe -i - -y output_data/videos/9xx-no-align.mp4
# for body in $(seq 900 907)
# do
#     ffmpeg -framerate 100 -i output_data/saved_images/test_$(echo $body)_00%3d.png -y output_data/videos/test_$body.mp4
# done

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
