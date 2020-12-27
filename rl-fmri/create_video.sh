#!/bin/sh

ffmpeg -framerate 20 -i exp_0/fMRI_videos/getCameraImage_b0_s0_00%3d.png -y exp_0/getCameraImage_b0_s0.mp4
ffmpeg -framerate 20 -i exp_0/fMRI_videos/getCameraImage_b100_s0_00%3d.png -y exp_0/getCameraImage_b100_s0.mp4
ffmpeg -framerate 20 -i exp_0/fMRI_videos/barchart_b0_s0_00%3d.png -y exp_0/barchart_b0_s0.mp4
ffmpeg -framerate 20 -i exp_0/fMRI_videos/barchart_b100_s0_00%3d.png -y exp_0/barchart_b100_s0.mp4

ffmpeg -i exp_0/getCameraImage_b0_s0.mp4 -vf "movie=exp_0/barchart_b0_s0.mp4, scale=500: -1 [inner];[in][inner]overlay=10:10 [out]" -y exp_0/b0_s0.mp4
ffmpeg -i exp_0/getCameraImage_b100_s0.mp4 -vf "movie=exp_0/barchart_b100_s0.mp4, scale=500: -1 [inner];[in][inner]overlay=10:10 [out]" -y exp_0/b100_s0.mp4
