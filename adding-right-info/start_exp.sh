#!/bin/sh
set -x

for s1 in 0 1 2
do
    sbatch -J $1 submit_train.sh "$1 --body-id $s1"
    sleep 0

    for s2 in 0 1 2
    do
        if [ "$s1" -lt "$s2" ]; then
            echo $s1 $s2
            sbatch -J $1 submit_train.sh "$1 --train-on-both-bodies --train-bodies $s1 $s2"
            sleep 0
            sbatch -J $1 submit_train.sh "$1 --with-bodyinfo --train-on-both-bodies --train-bodies $s1 $s2"
            sleep 0
        fi
    done
done
