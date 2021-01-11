#!/bin/sh

body=320

# generate videos
ffmpeg -framerate 100 -i output_data/saved_images/getCameraImage_00%3d.png -y output_data/videos/getCameraImage_$body.mp4
ffmpeg -framerate 100 -i output_data/saved_images/barchart_00%3d.png -y output_data/videos/barchart_$body.mp4

ffmpeg -i output_data/videos/getCameraImage_$body.mp4 -vf "movie=output_data/videos/barchart_$body.mp4, scale=500: -1 [inner];[in][inner]overlay=10:10 [out]" -y output_data/videos/$body.mp4


