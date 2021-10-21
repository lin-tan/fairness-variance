#!/bin/bash

CONFIG=/home/userQQQ/dlfairness/balanced_dataset_not_enough_coco.json
RAW_RESULT_DIR=/home/userQQQ/fairness_raw_result/result_balanced_dataset_coco
OUTPUT_DIR=/local2/userQQQ/fairness_neurips2021/weights

python bdne.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR