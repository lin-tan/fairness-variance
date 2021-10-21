#!/bin/bash
RESULT_DIR='/dlfairness/result_fairalm'
json_filename="fair_alm.json"

now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE=$RESULT_DIR"/analysis_$now.log"

exec &>$LOG_FILE

python 1_3_1_analyze_the_runs_fairness.py $RESULT_DIR $json_filename
