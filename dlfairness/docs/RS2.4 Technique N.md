# RS2: Technique "N"

The code for "N" technique is under `other/prediction_result/nifr`.

To the the model accuracy and prediction results, change the following variables in `run_celeba.sh` and `run_model_acc.sh` under `other/prediction_result/nifr`:

* `CONFIG`: The full path of `json_filename`. For example, `/dlfairness/nifr.json`
* `RAW_RESULT_DIR`: Use `RESULT_DIR`
* `OUTPUT_DIR`: Fill in the  prepared `OUTPUT_DIR` 

```bash
bash run_celea.sh
bash run_model_acc.sh
```

## Results

After running the two scripts, you will have the model accuracy for all the 16 FIT runs in a yaml file `other/prediction_result/nifr/celeba_nifr_acc.yaml` and raw prediction results under `OUTPUT_DIR`.

The `OUTPUT_DIR` will have similar structure like this:

```bash
n_aggregated_raw_prediction
├── baseline	# N-Base
└── inn			# N-Flow
```

Each csv file has four columns:

* `idx`: The index of the sample.
* `ground_truth`: The ground truth (task) label.
* `prediction_result`: The predicted (task) label of the model.
* `protected_label`: The protected label of the sample.

The results of our experiments can be found at: https://github.com/lin-tan/fairness-variance/releases/tag/prediction



The yaml file `celeba_nifr_acc.yaml` is a dictionary with the following keys:

| Technique | Key in`celeba_nifr_acc.yaml` |
| --------- | ---------------------------- |
| N-Base    | baseline/                    |
| N-Flow    | inn/                         |

The value to each key is a list containing the accuracies from 16 FIT runs.