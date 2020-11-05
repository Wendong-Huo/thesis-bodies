#!/bin/sh
set -x

for s1 in {80..99}
do
    sbatch submit_exp.sh $s1
    sleep 3
done
