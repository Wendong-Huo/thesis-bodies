#!/bin/sh
set -x

for s1 in {0..9}
do
    for s2 in {0..2}
    do
        sbatch submit_exp_3xg10.sh $s1 $s2
        sleep 3
    done
done
