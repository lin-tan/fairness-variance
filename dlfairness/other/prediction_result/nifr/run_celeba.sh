#!/bin/bash

CONFIG=/dlfairness/nifr.json
RAW_RESULT_DIR=/working_dir/result_n
OUTPUT_DIR=/working_dir/result_n/n_aggregated_raw_prediction

python celeba.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR