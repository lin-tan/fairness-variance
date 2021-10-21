#!/bin/bash
NAS_DIR='/dlfairness/result_fairalm'
done_filename="modify_done.csv"
json_filename="fair_alm.json"

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$NAS_DIR"/modify_training_files_$now.log"

exec &>$LOG_FILE

python 0_1_training_code_modify.py $NAS_DIR $done_filename $json_filename
