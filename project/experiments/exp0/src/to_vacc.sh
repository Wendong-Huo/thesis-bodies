#!/bin/sh
set -x

# ssh sliu1@vacc-user1.uvm.edu mkdir -p thesis-bodies-2021/experiments/$(cat .exp_folder)/src
EXP_FOLDER=$(cat .exp_folder)
rsync -r --rsync-path="mkdir -p thesis-bodies-2021/experiments/$EXP_FOLDER/src && rsync" --exclude-from=.gitignore .. sliu1@vacc-user1.uvm.edu:~/thesis-bodies-2021/experiments/$EXP_FOLDER
