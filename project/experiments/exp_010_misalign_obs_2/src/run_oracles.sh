#!/bin/sh
source activate thesis-bodies

set -x

python 0.init.py
EXP_FOLDER=$(cat .exp_folder)

for seed in 0 1
do
    for body in 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319
    do
        # Control
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$body --test_bodies=$body --train_steps=5e6
    done
    for body in 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419
    do
        # Control
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$body --test_bodies=$body --train_steps=5e6
    done
    for body in 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519
    do
        # Control
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$body --test_bodies=$body --train_steps=5e6
    done
    for body in 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619
    do
        # Control
        sbatch -J $EXP_FOLDER submit.sh python 1.train.py --seed=$seed --train_bodies=$body --test_bodies=$body --train_steps=5e6
    done
done

