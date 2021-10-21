#!/bin/bash

CONFIG=/dlfairness/fair_alm.json
RAW_RESULT_DIR=/working_dir/result_a
OUTPUT_DIR=/working_dir/result_a/a_aggregated_raw_prediction

python celeba_acc.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR