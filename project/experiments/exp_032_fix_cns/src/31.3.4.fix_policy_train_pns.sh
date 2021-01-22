#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
# ========================================

exp_name="3step"

description="Step1. train on Walker2D(399), because it's the hardest problem; Step2. continue train on V4, until all bodies are solved; Step3. continue train on V4, with CNS fixed."

