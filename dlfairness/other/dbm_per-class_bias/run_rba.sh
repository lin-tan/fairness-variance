#!/bin/bash

CONFIG=/home/userQQQ/dlfairness/fairness_variance_test_config.json
RAW_RESULT_DIR=/home/userQQQ/fairness_raw_result/DomainBiasMitigation_all_32runs_fixedseed1

python batch_rba.py --config $CONFIG --raw_result_dir $RAW_RESULT_DIR