#!/bin/sh

# clean
rm $1/tb -r
rm $1/fMRI_videos/*
rm $1/obs_data_videos/*
rm $1/*.npy

# generate jpgs
python test_obs_data.py --exp=$1 --train-bodies=300,0,300,300 --test-bodies=0,300 --disable-wrapper --save-obs-data --render --test-steps=1000

# generate videos
ffmpeg -framerate 20 -i $1/obs_data_videos/getCameraImage_b0_s0_00%3d.png -y $1/getCameraImage_b0_s0.mp4
ffmpeg -framerate 20 -i $1/obs_data_videos/getCameraImage_b300_s0_00%3d.png -y $1/getCameraImage_b300_s0.mp4
ffmpeg -framerate 20 -i $1/obs_data_videos/barchart_b0_s0_00%3d.png -y $1/barchart_b0_s0.mp4
ffmpeg -framerate 20 -i $1/obs_data_videos/barchart_b300_s0_00%3d.png -y $1/barchart_b300_s0.mp4

ffmpeg -i $1/getCameraImage_b0_s0.mp4 -vf "movie=$1/barchart_b0_s0.mp4, scale=500: -1 [inner];[in][inner]overlay=10:10 [out]" -y $1/b0_s0.mp4
ffmpeg -i $1/getCameraImage_b300_s0.mp4 -vf "movie=$1/barchart_b300_s0.mp4, scale=500: -1 [inner];[in][inner]overlay=10:10 [out]" -y $1/b300_s0.mp4
