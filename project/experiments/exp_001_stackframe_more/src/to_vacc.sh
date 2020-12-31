#!/bin/sh
set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

# ssh sliu1@vacc-user1.uvm.edu mkdir -p thesis-bodies-2021/experiments/$(cat .exp_folder)/src
rsync -r --rsync-path="mkdir -p thesis-bodies-2021/experiments/$EXP_FOLDER && rsync" --exclude-from=.gitignore .. sliu1@vacc-user1.uvm.edu:~/thesis-bodies-2021/experiments/$EXP_FOLDER
