#!/bin/sh
set -x

for s1 in {0..4}
do
    for s2 in {0..2}
    do
        sbatch submit_exp_3xg20_bodyinfo.sh $s1 $s2
        sbatch submit_exp_3xg20_nobodyinfo.sh $s1 $s2
        sleep 3
    done
done
