# RS2: Technique "C" and "I"

The code for "C" and "I" techniques is under `other/prediction_result/balanced_dataset_csv_pred_result`.

To the the model accuracy and prediction results, change the following variables in `run_coco.sh`, `run_coco_adv.sh`, `run_imSitu.sh`, `run_imSitu_adv.sh`, and `run_model_acc.sh` under `other/prediction_result/balanced_dataset_csv_pred_result`:

* `CONFIG`: The full path of `json_filename`. For example, `/dlfairness/balanced_dataset_not_enough_coco.json`
* `RAW_RESULT_DIR`: Use `RESULT_DIR`. For example, `/working_dir/result_c`
* `OUTPUT_DIR`: Fill in the  prepared `OUTPUT_DIR`. For example, `/working_dir/result_c/c_aggregated_raw_prediction` (Technique "C") and `/working_dir/result_c/i_aggregated_raw_prediction` (Technique "I")

```bash
bash run_coco.sh
bash run_coco_adv.sh
bash run_imSitu.sh
bash run_imSitu_adv.sh
bash run_model_acc.sh
```

## Results

After running the two scripts, you will have the model accuracy for all the 16 FIT runs in two yaml files `other/prediction_result/balanced_dataset_csv_pred_result/coco_mAP.yaml` and `other/prediction_result/balanced_dataset_csv_pred_result/imSitu_mAP.yaml`. The raw prediction results are under `OUTPUT_DIR`.

The `OUTPUT_DIR` will have similar structure like this:

```bash
c_aggregated_raw_prediction
...
├── threshold_0.5
│   ├── adv-conv4	# C-A4
│   ├── adv-conv5	# C-A5
│   ├── no_gender	# C-Base
│   ├── ratio-1		# C-R1
│   ├── ratio-2		# C-R2
│   └── ratio-3		# C-R3
...

i_aggregated_raw_prediction
├── adv-conv4		# I-A4
├── adv-conv5		# I-A5
├── no_gender		# I-Base
├── ratio-1			# I-R1
├── ratio-2			# I-R2
└── ratio-3			# I-R3
```

For MS-COCO dataset, we only focus the files under `threshold_0.5` as suggested by the original paper. However, you will still need to copy all the files under `OUTPUT_DIR` in [RS3](#rs3-bias-metric-calculation).

Each csv file has four columns:

* `idx`: The index of the sample.
* `ground_truth`: The ground truth (task) label.
* `prediction_result`: The predicted (task) label of the model.
* `protected_label`: The protected label of the sample.

The results of our experiments can be found at: https://github.com/lin-tan/fairness-variance/releases/tag/prediction



The yaml file `coco_mAP.yaml` and `imSitu_mAP.yaml` are dictionaries with the following keys:

| Technique     | Key in`coco_mAP.yaml`        |
| ------------- | ---------------------------- |
| C-Base        | threshold_0.5/no_gender/     |
| C-R1          | threshold_0.5/ratio-1/       |
| C-R2          | threshold_0.5/ratio-2/       |
| C-R3          | threshold_0.5/ratio-3/       |
| C-A4          | threshold_0.5/adv-conv4/     |
| C-A5          | threshold_0.5/adv-conv5/     |
| **Technique** | **Key in `imSitu_mAP.yaml`** |
| I-Base        | no_gender/                   |
| I-R1          | ratio-1/                     |
| I-R2          | ratio-2/                     |
| I-R3          | ratio-3/                     |
| I-A4          | adv-conv4/                   |
| I-A5          | adv-conv5/                   |

The value to each key is a list containing the accuracies from 16 FIT runs.