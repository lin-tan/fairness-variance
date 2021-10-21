#!/bin/bash
RESULT_DIR='/dlfairness/result_fairalm'
NAS_DIR='/dlfairness/result_fairalm'
done_filename="train_done.csv"
queue_filename="queue.txt"

server="neurips2021"

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$RESULT_DIR"/train_per_epoch_$now.log"

exec &>$LOG_FILE

python 1_2_execute_server_runs.py $RESULT_DIR $NAS_DIR $done_filename $queue_filename $server
