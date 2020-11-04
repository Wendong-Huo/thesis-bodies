#!/bin/sh

for s1 in 0.3 0.6 1.0 1.2 1.5 1.8
do
    for s2 in 0.3 0.6 1.0 1.2 1.5 1.8
    do
        for s3 in 0.3 0.6 1.0 1.2 1.5 1.8
        do
# for s1 in 1.0 1.2
# do
#     for s2 in 1.0 
#     do
#         for s3 in 1.0 1.2 
#         do
            sbatch submit_exp.sh $s1 $s2 $s3
            sleep 3
        done
    done
done
