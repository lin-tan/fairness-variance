#!/bin/bash
NAS_DIR='/home/userHHH/Workspace/dlvariancetesting/result'
done_filename="train_done.csv"
queue_filename="queue.txt"
json_filename="variance_test_config.json"

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$NAS_DIR"/setup_train_$now.log"

exec &>$LOG_FILE

python 1_1_setup_server_runs.py $NAS_DIR $done_filename $queue_filename $json_filename
