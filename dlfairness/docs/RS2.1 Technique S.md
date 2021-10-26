# RS2: Technique "S"

The code for "S" techniques is under `other/prediction_result/dbm_csv_pred_result`.

To the the model accuracy and prediction results, change the following variables in `run.sh` and `run_acc.sh` under `other/prediction_result/dbm_csv_pred_result`:

* `CONFIG`: The full path of `json_filename`. For example, `/dlfairness/fairness_variance_test_config.json`
* `RAW_RESULT_DIR`: Use `RESULT_DIR`. For example, `/working_dir/result_s`
* `OUTPUT_DIR`: Fill in the  prepared `OUTPUT_DIR`. For example, `/working_dir/result_s/s_aggregated_raw_prediction`

```bash
bash run.sh
bash run_acc.sh
```

## Results

After running the two scripts, you will have the model accuracy for all the 16 FIT runs in a yaml file `other/prediction_result/dbm_csv_pred_result/cifar-10s.yaml` and the raw prediction results under `OUTPUT_DIR`.

The `OUTPUT_DIR` will have similar structure like this:

```bash
s_aggregated_raw_prediction
├── baseline	# S-Base
│   ├── try_00.csv
│   ├── try_01.csv
...
│   └── try_15.csv
├── domain-discriminative
│   ├── Max-of-prob-prior-shift	# S-DD2
│   │   ├── try_00.csv
│   │   ├── try_01.csv
...
│   │   └── try_15.csv
│   ├── rba	# S-DD4
│   │   ├── try_00.csv
...
│   │   └── try_15.csv
│   ├── Sum-of-prob-no-prior-shift # S-DD1
│   │   ├── try_00.csv
...
│   │   └── try_15.csv
│   └── Sum-of-prob-prior-shift	# S-DD3
│       ├── try_00.csv
...
│       └── try_15.csv
├── domain-independent
│   ├── Conditional	# S-DI1
│   │   ├── try_00.csv
...
│   │   └── try_15.csv
│   └── Sum	# S-DI2
│       ├── try_00.csv
...
│       └── try_15.csv
├── gradproj-adv	# S-GR
│   ├── try_00.csv
...
│   └── try_15.csv
├── sampling	# S-RS
│   ├── try_00.csv
...
│   └── try_15.csv
└── uniconf-adv	# S-UC
    ├── try_00.csv
...
    └── try_15.csv
```

Each csv file has four columns:
* `idx`: The index of the sample.
* `ground_truth`: The ground truth (task) label.
* `prediction_result`: The predicted (task) label of the model.
* `protected_label`: The protected label of the sample.

The results of our experiments can be found at: https://github.com/lin-tan/fairness-variance/releases/tag/prediction



The yaml file `cifar-10s.yaml` is a dictionary with the following keys:

| Technique | Key in `cifar-10s.yaml`                           |
| --------- | ------------------------------------------------- |
| S-Base    | baseline/                                         |
| S-RS      | sampling/                                         |
| S-UC      | uniconf-adv/                                      |
| S-GR      | gradproj-adv/                                     |
| S-DD1     | domain-discriminative/Sum-of-prob-no-prior-shift/ |
| S-DD2     | domain-discriminative/Max-of-prob-prior-shift/    |
| S-DD3     | domain-discriminative/Sum-of-prob-prior-shift/    |
| S-DD4     | domain-discriminative/rba/                        |
| S-DI1     | domain-independent/Conditional/                   |
| S-DI2     | domain-independent/Sum/                           |

The value to each key is a list containing the accuracies from 16 FIT runs.
