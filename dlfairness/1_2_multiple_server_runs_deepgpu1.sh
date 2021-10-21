#!/bin/bash
RESULT_DIR='/local/userHHH/Workspace/dlvariancetesting/result_adv'
NAS_DIR='/home/userHHH/Workspace/dlvariancetesting/result_adv'
done_filename="train_done.csv"
queue_filename="adv_queue.txt"

server="deepgpu1"

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$RESULT_DIR"/train_per_epoch_$now.log"

exec &>$LOG_FILE

python 1_2_execute_server_runs.py $RESULT_DIR $NAS_DIR $done_filename $queue_filename $server
