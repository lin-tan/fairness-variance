#!/bin/bash
RESULT_DIR='result'
json_filename="variance_test_config.json"

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$RESULT_DIR"/analysis_$now.log"

exec &>$LOG_FILE

python 1_3_analyze_the_runs.py $RESULT_DIR $json_filename
