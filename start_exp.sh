#!/bin/sh

for seed in 10 11 12
do
    sbatch submit_exp.sh $seed
    sleep 3
done
