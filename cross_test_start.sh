#!/bin/sh
set -x

for s1 in {0..9}
do
    sbatch cross_test.sh $s1
done
