#!/bin/sh
set -x

hostname -I

tensorboard --logdir output_data/tensorboard --host 0.0.0.0