#!/bin/bash
RESULT_DIR='result'
run_configs_file="variance_test_config.json"
comparison_configs_file="comparing_config.json"

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$RESULT_DIR"/comparing_$now.log"

exec &>$LOG_FILE

python 1_4_compare_runs.py $RESULT_DIR $run_configs_file $comparison_configs_file
