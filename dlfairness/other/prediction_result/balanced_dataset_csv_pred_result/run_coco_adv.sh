#!/bin/bash

CONFIG=/dlfairness/balanced_dataset_not_enough_coco.json
RAW_RESULT_DIR=/working_dir/result_c
OUTPUT_DIR=/working_dir/result_c/c_aggregated_raw_prediction

python coco_adv.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR --threshold 0.1
python coco_adv.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR --threshold 0.2
python coco_adv.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR --threshold 0.3
python coco_adv.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR --threshold 0.4
python coco_adv.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR --threshold 0.5
python coco_adv.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR --threshold 0.6
python coco_adv.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR --threshold 0.7
python coco_adv.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR --threshold 0.8
python coco_adv.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR --output_dir $OUTPUT_DIR --threshold 0.9