#!/usr/bin/env python3

import glob
import os
import pprint
import traceback

import click
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from common.tflogs2pandas import tflog2pandas

df = tflog2pandas("output_data/tensorboard/model-400-stack4-sd4/PPO_1")
print(df)
