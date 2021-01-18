#!/bin/sh

set -x

for body in $(seq 101 119)
do
    python 3.save_images_for_video.py --test_bodies=$body --model_filename output_data/models/model-$body-sd0.zip --render
done