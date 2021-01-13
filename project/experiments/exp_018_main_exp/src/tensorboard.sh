#!/bin/sh
set -x

hostname -I

tensorboard --logdir output_data/${1:-tensorboard} --host 0.0.0.0