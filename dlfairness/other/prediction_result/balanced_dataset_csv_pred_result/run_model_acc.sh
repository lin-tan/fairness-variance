#!/bin/bash

CONFIG=/dlfairness/balanced_dataset_not_enough_coco.json
RAW_RESULT_DIR=/working_dir/result_c
OUTPUT_DIR=/working_dir/result_c/c_aggregated_raw_prediction

python model_acc.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR