#!/bin/bash

CONFIG=/home/userQQQ/dlfairness/fair_alm.json
RAW_RESULT_DIR=/home/userQQQ/fairness_raw_result/result_fairalm
OUTPUT_DIR=/local2/userQQQ/fairness_neurips2021/weights

python alm.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR