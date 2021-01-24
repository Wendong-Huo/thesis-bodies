#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="ChannelSize2"

description="Just implemented new PNSCNS policy, train on Ants and test channel size."

# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=4  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/4_4 --seed=8080
# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=4  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/4_4 --seed=80801

# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=8  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/8_8 --seed=8080
# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=8  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/8_8 --seed=80801

# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=16 --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/16_8 --seed=8080
# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=8  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/16_8 --seed=80801

# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=32 --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/32_32 --seed=8080
# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=32 --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/32_32 --seed=80801

# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=64 --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/64_64 --seed=8080
# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=64 --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/64_64 --seed=80801

# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=128 --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/128_128 --seed=8080
# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=128 --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/128_128 --seed=80801

# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=256 --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/256_256 --seed=8080
# sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=256 --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/channel_size/256_256 --seed=80801

sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=32 --cnspns_motor_channel=32  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/$exp_name/32_32 --seed=8082
sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=32 --cnspns_motor_channel=32  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/$exp_name/32_32 --seed=8083

sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=16 --cnspns_motor_channel=16  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/$exp_name/16_16 --seed=8082
sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=16 --cnspns_motor_channel=16  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/$exp_name/16_16 --seed=8083

sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=8 --cnspns_motor_channel=8  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/$exp_name/8_8 --seed=8082
sbatch -J $exp_name $submit_to python 1.train.py --cnspns_sensor_channel=8 --cnspns_motor_channel=8  --train_bodies=599 --test_bodies=599 --cnspns --tensorboard=tensorboard/$exp_name/8_8 --seed=8083

# ========================================
# log
echo "================" >> ~/gpfs2/experiments.log
date >> ~/gpfs2/experiments.log
pwd >> ~/gpfs2/experiments.log
echo $0 >> ~/gpfs2/experiments.log
echo $exp_name >> ~/gpfs2/experiments.log
echo $description >> ~/gpfs2/experiments.log
squeue -O "JobID,Partition,Name,Nodelist,TimeUsed,UserName,StartTime,Schednodes,Command" --user=sliu1 -n $exp_name | head -n 10 >> ~/gpfs2/experiments.log
echo "================" >> ~/gpfs2/experiments.log

4-show-experiment-log.sh
