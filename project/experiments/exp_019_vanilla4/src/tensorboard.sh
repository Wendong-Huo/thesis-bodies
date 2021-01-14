#!/bin/sh
set -x

hostname -I

tensorboard --logdir ${1:-output_data/tensorboard} --host 0.0.0.0