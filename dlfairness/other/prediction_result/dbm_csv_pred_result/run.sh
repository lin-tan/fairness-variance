#!/bin/bash

CONFIG=/dlfairness/fairness_variance_test_config.json
RAW_RESULT_DIR=/working_dir/result_s
OUTPUT_DIR=/working_dir/result_s/s_aggregated_raw_prediction

python main.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR