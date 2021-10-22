# Reproducibility Guide for "Are My Deep Learning Systems Fair? An Empirical Study of Fixed-Seed Training"

## Introduction

This is the replication package for paper "Are My Deep Learning Systems Fair? An Empirical Study of Fixed-Seed Training".

To simplify the path used in this documentation, we assume that the code is stored under `/dlfairness`, and all the paths mentioned without a `/` at the beginning is relative to `/dlfairness`.

**Note: The results from 16 fixed-seed identical training runs (FIT runs) in our paper are a sampling of the real-world distribution of fairness variance observed. Even with the same code, one might get slightly different results than what were reported in the paper.**



## System Requirements

Python 3.6+ w/ pip installed

Docker-CE 19.03+

Packages inside `requirements.txt` (`pip3 install -r requirements.txt`)

Two [docker images](https://github.com/lin-tan/fairness-variance/releases/tag/docker_image) provided (`docker load -i dlfairness_balanced_dataset_not_enough_coco.tar` and `docker load -i dlfairness_nifr.tar`)

**We strongly recommend running the experiments under a separate Python Virtual Environment (virtualenv).**



## Overview

All the results in the paper can be replicated with five steps (**RS**: Reproduce Step):

1. Prepare the dataset for each reproduced paper ([RS1](#rs1-executing-code-for-reproduced-papers))
2. Execute the code for each reproduced paper ([RS1](#rs1-executing-code-for-reproduced-papers))
3. Collect the prediction results on the testset ([RS2](#rs2-collecting-prediction-results))
4. Calculate the bias metrics ([RS3](#rs3-bias-metric-calculation))
5. Generate the results from the bias values calculated ([RS4](#rs4-generate-final-results))



One json config file is associated with each experiment. To run the experiment, we need five paths:

* `shared_dir`: Status directory for all the reproduced papers
* `working_dir`: Working directory for all the reproduced papers
* `modified_target_dir`: Intermediate directory before the execution of all the reproduced papers
* `NAS_DIR`: A directory under `shared_dir` to store the status file for each reproduced paper (`NAS_DIR` = `shared_dir/result_<paper>`)
* `RESULT_DIR`: A directory under `working_dir` as the working directory for each reproduced paper (`RESULT_DIR` = `working_dir/result_<paper>`)

We refer to the json config file as `json_filename`.



### Path Samples

We provide the following sample set of path to reproduce our study.

| Technique | json_filename                         | shared_dir  | working_dir  | modified_target_dir | NAS_DIR              | RESULT_DIR            |
| --------- | ------------------------------------- | ----------- | ------------ | ------------------- | -------------------- | --------------------- |
| S         | fairness_variance_test_config.json    | /dlfairness | /working_dir | N/A                 | /dlfairness/result_s | /working_dir/result_s |
| C, I      | balanced_dataset_not_enough_coco.json | /dlfairness | /working_dir | /modified_dir       | /dlfairness/result_c | /working_dir/result_c |
| A         | fair_alm.json                         | /dlfairness | /working_dir | /modified_dir       | /dlfairness/result_a | /working_dir/result_a |
| N         | nifr.json                             | /dlfairness | /working_dir | /modified_dir       | /dlfairness/result_n | /working_dir/result_n |

All four json config files can be found under `/dlfairness`. Column _Technique_ is the initials for tasks in our paper.



## RS1. Executing Code for Reproduced Papers

### What you need for this step

**Define** `shared_dir`, `working_dir`, `modified_target_dir`, `NAS_DIR`, and `RESULT_DIR` for all four reproduced papers



### Technique "S"

**Acknowledgement: The code is adapted from https://github.com/princetonvisualai/DomainBiasMitigation**

The code is stored under `original_code/DomainBiasMitigation`.

#### Prepare the dataset

```bash
cd original_code/DomainBiasMitigation
bash download.sh
python3 preprocess_data.py
```

#### Prepare the paths

Open `fairness_variance_test_config.json` and fill in all the entries with keys `shared_dir` and `working_dir` with the paths you choose.

Open `0_1_1_training_code_modify_fairness.sh`, modify `NAS_DIR` on Line 2 and `json_filename` on Line 4.

Open `1_1_1_setup_server_runs_fairness.sh`, modify `NAS_DIR` on Line 2 and `json_filename` on Line 5.

Open `1_2_1_multiple_server_runs_deepgpu3_fairness.sh`, modify `RESULT_DIR` on Line 2 and `NAS_DIR` on Line 3.

#### Run the scripts

```bash
bash 0_1_1_training_code_modify_fairness.sh
bash 1_1_1_setup_server_runs_fairness.sh
bash 1_2_1_multiple_server_runs_deepgpu3_fairness.sh
```

#### Results

After running the scripts, you will find similar files and directories under your `RESULT_DIR`.

```bash
.
├── ResNet-18_baseline_cifar-s_fixed
│   ├── 0
│   ├── 1
...
│   ├── 14
│   └── 15
├── ResNet-18_domain-discriminative_cifar-s_fixed
...
├── ResNet-18_domain-independent_cifar-s_fixed
...
├── ResNet-18_gradproj-adv_cifar-s_fixed
...
├── ResNet-18_sampling_cifar-s_fixed
...
├── ResNet-18_uniconf-adv_cifar-s_fixed
...
└── train_per_epoch_20_11_02_21_42_45.log
```



### Technique "C" and "I"

**Acknowledgement: The code is adapted from https://github.com/uvavision/Balanced-Datasets-Are-Not-Enough**

The code is stored under `original_code/Balanced-Datasets-Are-Not-Enough/object_multilabel` (technique C) and `original_code/Balanced-Datasets-Are-Not-Enough/verb_classification` (technique I).

#### Prepare the dataset

```bash
# MS-COCO
cd original_code/Balanced-Datasets-Are-Not-Enough/object_multilabel/data
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip train2014.zip
unzip val2014.zip
unzip test2014.zip
unzip annotations_trainval2014.zip
mv annotations annotations_pytorch

# imSitu
cd original_code/Balanced-Datasets-Are-Not-Enough/verb_classification/data
wget https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar
tar xvf of500_images_resized.tar
```

#### Prepare the paths

Open `balanced_dataset_not_enough_coco.json` and fill in all the entries with keys `shared_dir`, `working_dir`, and `modified_target_dir` with the paths you choose.

Open `0_1_1_training_code_modify_fairness.sh`, modify `NAS_DIR` on Line 2 and `json_filename` on Line 4.

Open `1_1_1_setup_server_runs_fairness.sh`, modify `NAS_DIR` on Line 2 and `json_filename` on Line 5.

Open `1_2_1_multiple_server_runs_deepgpu3_fairness.sh`, modify `RESULT_DIR` on Line 2 and `NAS_DIR` on Line 3.

#### Run the scripts

```bash
bash 0_1_1_training_code_modify_fairness.sh
bash 1_1_1_setup_server_runs_fairness.sh
bash 1_2_1_multiple_server_runs_deepgpu3_fairness.sh
```

#### Results

After running the scripts, you will find similar files and directories under your `RESULT_DIR`.

```bash
.
├── BDNE-ResNet-50_adv-conv4_coco_fixed1
├── BDNE-ResNet-50_adv-conv4_imSitu_fixed1
├── BDNE-ResNet-50_adv-conv5_coco_fixed1
├── BDNE-ResNet-50_adv-conv5_imSitu_fixed1
├── BDNE-ResNet-50_no_gender_coco_fixed1
├── BDNE-ResNet-50_no_gender_imSitu_fixed1
├── BDNE-ResNet-50_ratio-1_coco_fixed1
├── BDNE-ResNet-50_ratio-1_imSitu_fixed1
├── BDNE-ResNet-50_ratio-2_coco_fixed1
├── BDNE-ResNet-50_ratio-2_imSitu_fixed1
├── BDNE-ResNet-50_ratio-3_coco_fixed1
├── BDNE-ResNet-50_ratio-3_imSitu_fixed1
├── BDNE-ResNet-50_test1_imSitu_fixed1
├── BDNE-ResNet-50_test2_imSitu_fixed1
└── train_per_epoch_21_01_11_17_44_43.log
```



### Technique "A"

**Acknowledgement: The code is adapted from https://github.com/lokhande-vishnu/FairALM**

The code is stored under `original_code/FairALM/Experiments-CelebA`.

#### Prepare the dataset

```bash
# CelebA
cd original_code/FairALM/Experiments-CelebA/data
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/img_align_celeba.zip
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/list_attr_celeba.txt
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/list_eval_partition.txt
unzip img_align_celeba.zip
```

#### Prepare the paths

Open `fair_alm.json` and fill in all the entries with keys `shared_dir`, `working_dir`, and `modified_target_dir` with the paths you choose.

Open `0_1_1_training_code_modify_fairness.sh`, modify `NAS_DIR` on Line 2 and `json_filename` on Line 4.

Open `1_1_1_setup_server_runs_fairness.sh`, modify `NAS_DIR` on Line 2 and `json_filename` on Line 5.

Open `1_2_1_multiple_server_runs_deepgpu3_fairness.sh`, modify `RESULT_DIR` on Line 2 and `NAS_DIR` on Line 3.

#### Run the scripts

```bash
bash 0_1_1_training_code_modify_fairness.sh
bash 1_1_1_setup_server_runs_fairness.sh
bash 1_2_1_multiple_server_runs_deepgpu3_fairness.sh
```

#### Results

After running the scripts, you will find similar files and directories under your `RESULT_DIR`.

```bash
.
├── ResNet18_fair-alm_CelebA_fixed1
├── ResNet18_fair-alm_CelebA_fixed1_variance_analysis.csv
├── ResNet18_l2-penalty_CelebA_fixed1
├── ResNet18_l2-penalty_CelebA_fixed1_variance_analysis.csv
├── ResNet18_no-constraints_CelebA_fixed1
├── ResNet18_no-constraints_CelebA_fixed1_variance_analysis.csv
└── train_per_epoch_21_02_14_15_03_05.log
```



### Technique "N"

**Acknowledgement: The code is adapted from https://github.com/predictive-analytics-lab/nifr**

The code is stored under `original_code/nifr`.

#### Prepare the dataset

```bash
# CelebA
cd original_code/nifr/data/celeba
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/img_align_celeba.zip
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/list_attr_celeba.txt
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/list_eval_partition.txt
unzip img_align_celeba.zip
```

#### Prepare the paths

Open `nifr.json` and fill in all the entries with keys `shared_dir`, `working_dir`, and `modified_target_dir` with the paths you choose.

Open `0_1_1_training_code_modify_fairness.sh`, modify `NAS_DIR` on Line 2 and `json_filename` on Line 4.

Open `1_1_1_setup_server_runs_fairness.sh`, modify `NAS_DIR` on Line 2 and `json_filename` on Line 5.

Open `1_2_1_multiple_server_runs_deepgpu3_fairness.sh`, modify `RESULT_DIR` on Line 2 and `NAS_DIR` on Line 3.

#### Run the scripts

```bash
bash 0_1_1_training_code_modify_fairness.sh
bash 1_1_1_setup_server_runs_fairness.sh
bash 1_2_1_multiple_server_runs_deepgpu3_fairness.sh
```

#### Results

After running the scripts, you will find similar files and directories under your `RESULT_DIR`.

```bash
.
├── NIFR_baseline_CelebA_fixed1
├── NIFR_inn_CelebA_fixed1
└── train_per_epoch_21_01_23_18_37_42.log
```



## RS2. Collecting Prediction Results

### What you need for this step

**From [RS1](#what-you-need-for-this-step)**: `shared_dir`, `working_dir`, `modified_target_dir`, `NAS_DIR`, and `RESULT_DIR` for all four reproduced papers

**Define** `OUTPUT_DIR` for each set of techniques with the same initial



### Overview

In this step, we will collect the prediction results of the models trained in each FIT run as well as collecting the model accuracy data. In addition of the five paths mentioned above (`shared_dir`, `working_dir`, `modified_target_dir`, `NAS_DIR`, and `RESULT_DIR`), we will need another path `OUTPUT_DIR` to store all the prediction results.

| Technique | RESULT_DIR            | OUTPUT_DIR                                        |
| --------- | --------------------- | ------------------------------------------------- |
| S         | /working_dir/result_s | /working_dir/result_s/s_aggregated_raw_prediction |
| C         | /working_dir/result_c | /working_dir/result_c/c_aggregated_raw_prediction |
| I         | /working_dir/result_c | /working_dir/result_c/i_aggregated_raw_prediction |
| A         | /working_dir/result_a | /working_dir/result_a/a_aggregated_raw_prediction |
| N         | /working_dir/result_n | /working_dir/result_n/n_aggregated_raw_prediction |



### Technique "S"

The code for "S" techniques is under `other/prediction_result/dbm_csv_pred_result`.

To the the model accuracy and prediction results, change the following variables in `run.sh` and `run_acc.sh` under `other/prediction_result/dbm_csv_pred_result`:

* `CONFIG`: The full path of `json_filename`. For example, `/dlfairness/fairness_variance_test_config.json`
* `RAW_RESULT_DIR`: Use `RESULT_DIR`
* `OUTPUT_DIR`: Fill in the  prepared `OUTPUT_DIR` 

```bash
bash run.sh
bash run_acc.sh
```

#### Results

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



### Technique "C" and "I"

The code for "C" and "I" techniques is under `other/prediction_result/balanced_dataset_csv_pred_result`.

To the the model accuracy and prediction results, change the following variables in `run_coco.sh`, `run_coco_adv.sh`, `run_imSitu.sh`, `run_imSitu_adv.sh`, and `run_model_acc.sh` under `other/prediction_result/balanced_dataset_csv_pred_result`:

* `CONFIG`: The full path of `json_filename`. For example, `/dlfairness/balanced_dataset_not_enough_coco.json`
* `RAW_RESULT_DIR`: Use `RESULT_DIR`
* `OUTPUT_DIR`: Fill in the  prepared `OUTPUT_DIR` 

```bash
bash run_coco.sh
bash run_coco_adv.sh
bash run_imSitu.sh
bash run_imSitu_adv.sh
bash run_model_acc.sh
```

#### Results

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

For MS-COCO dataset, we only focus the files under `threshold_0.5` as suggested by the original paper. However, you will still need to copy all the files under `OUTPUT_DIR` in RS3.

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



### Technique "A"

The code for "A" technique is under `other/prediction_result/fair_alm`.

To the the model accuracy and prediction results, change the following variables in `run_celeba.sh` and `run_model_acc.sh` under `other/prediction_result/fair_alm`:

* `CONFIG`: The full path of `json_filename`. For example, `/dlfairness/fair_alm.json`
* `RAW_RESULT_DIR`: Use `RESULT_DIR`
* `OUTPUT_DIR`: Fill in the  prepared `OUTPUT_DIR` 

```bash
bash run_celea.sh
bash run_model_acc.sh
```

#### Results

After running the two scripts, you will have the model accuracy for all the 16 FIT runs in a yaml file `other/prediction_result/fair_alm/celeba_alm_acc.yaml` and raw prediction results under `OUTPUT_DIR`.

The `OUTPUT_DIR` will have similar structure like this:

```bash
a_aggregated_raw_prediction
├── fair-alm		# A-ALM
├── l2-penalty		# A-L2
└── no-constraints	# A-Base
```

Each csv file has four columns:

* `idx`: The index of the sample.
* `ground_truth`: The ground truth (task) label.
* `prediction_result`: The predicted (task) label of the model.
* `protected_label`: The protected label of the sample.

The results of our experiments can be found at: https://github.com/lin-tan/fairness-variance/releases/tag/prediction



The yaml file `celeba_alm_acc.yaml` is a dictionary with the following keys:

| Technique | Key in`celeba_alm_acc.yaml` |
| --------- | --------------------------- |
| A-Base    | no-constraints/             |
| A-L2      | l2-penalty/                 |
| A-ALM     | fair-alm/                   |

The value to each key is a list containing the accuracies from 16 FIT runs.



### Technique "N"

The code for "N" technique is under `other/prediction_result/nifr`.

To the the model accuracy and prediction results, change the following variables in `run_celeba.sh` and `run_model_acc.sh` under `other/prediction_result/nifr`:

* `CONFIG`: The full path of `json_filename`. For example, `/dlfairness/nifr.json`
* `RAW_RESULT_DIR`: Use `RESULT_DIR`
* `OUTPUT_DIR`: Fill in the  prepared `OUTPUT_DIR` 

```bash
bash run_celea.sh
bash run_model_acc.sh
```

#### Results

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



## RS3. Bias Metric Calculation

### What you need for this step

**From [RS2](#what-you-need-for-this-step-1)**: Contents under all the `OUTPUT_DIR`s in RS2, including the parent folder `OUTPUT_DIR`



### Overview

The code for bias metric calculation is self-contained and is located under `other/metric_calculation`. You will need the content of all the `OUTPUT_DIR`s in the previous section (RS2).

Details of bias metric calculation code can be found [here](other/metric_calculation/README.md).



**To calculate all the bias metrics, follow these steps:**

1. Copy all the `OUTPUT_DIR`s to `other/metric_calculation/prediction` (including the parent folder `OUTPUT_DIR`)

   After this step, the directory will have structure similar to this (only a subset of the files are listed):

   ```bash
   other/metric_calculation/prediction/
       |-- s_aggregated_raw_prediction
       |   |-- baseline
       |   |   |-- try_00.csv
       |   |   └-- try_01.csv
       |   └-- domain-discriminative
       |       |-- Sum-of-prob-no-prior-shift
       |       |   |-- try_00.csv
       |       |   └-- try_01.csv
       |       └-- rba
       |           |-- try_00.csv
       |           └-- try_01.csv
       └-- n_aggregated_raw_prediction
           |-- baseline
           |   |-- try_00.csv
           |   └-- try_01.csv
           └-- inn
               |-- try_00.csv
               └-- try_01.csv
   ...
   ```

2. Calculate the confusion matrix with `python3 confusion_matrix.py`

3. Calculate all the metrics with `bash calculate.sh`



Assume the `OUTPUT_DIR`s follow the example in the previous section (RS2), five folders will be created under `other/metric_calculation`:

* `other/metric_calculation/s_aggregated_raw_prediction`
* `other/metric_calculation/c_aggregated_raw_prediction`
* `other/metric_calculation/i_aggregated_raw_prediction`
* `other/metric_calculation/a_aggregated_raw_prediction`
* `other/metric_calculation/n_aggregated_raw_prediction`

And the structure will be similar to this:

```bash
metric_calculation/
    |-- s_aggregated_raw_prediction
    |   |-- cm
    |   |-- count
    |   └-- result
    |       |-- bias_amplification.csv
    |       |-- bias_amplification_raw.json
    |       |-- disparate_impact_factor.csv
    |       |-- disparate_impact_factor_raw.json
    |       |-- equality_of_odds_false_positive.csv
    |       |-- equality_of_odds_false_positive_raw.json
    |       |-- equality_of_odds_true_positive.csv
    |       |-- equality_of_odds_true_positive_raw.json
    |       |-- false_positive_subgroup_fairness.csv
    |       |-- false_positive_subgroup_fairness_raw.json
    |       |-- mean_difference_score.csv
    |       |-- mean_difference_score_raw.json
    |       |-- statistical_parity.csv
    |       └-- statistical_parity_raw.json
    └-- c_aggregated_raw_prediction`
        |-- cm
        |-- count
        └-- result
            |-- bias_amplification.csv
            |-- bias_amplification_raw.json
            |-- disparate_impact_factor.csv
            |-- disparate_impact_factor_raw.json
            |-- equality_of_odds_false_positive.csv
            |-- equality_of_odds_false_positive_raw.json
            |-- equality_of_odds_true_positive.csv
            |-- equality_of_odds_true_positive_raw.json
            |-- false_positive_subgroup_fairness.csv
            |-- false_positive_subgroup_fairness_raw.json
            |-- mean_difference_score.csv
            |-- mean_difference_score_raw.json
            |-- statistical_parity.csv
            └-- statistical_parity_raw.json
...
```

We will only focus on the `result` subdirectory under each of the five directories, and we will need the csv and json files for next step.



The csv and json files contains the bias values for all the experiments. A mapping of filename to metric name is as follows:

| Metric        | Filename                         |
| ------------- | -------------------------------- |
| DP            | mean_difference_score            |
| Normalized DI | disparate_impact_factor          |
| SPSF          | statistical_parity               |
| FPSF          | false_positive_subgroup_fairness |
| EOFP          | equality_of_odds_false_positive  |
| EOTP          | equality_of_odds_true_positive   |
| BA            | bias_amplification               |



## RS4. Generate Final Results

### What you need for this step

**From [RS2](#what-you-need-for-this-step-1)**: yaml files containing model accuracy:

* `other/prediction_result/dbm_csv_pred_result/cifar-10s.yaml`
* `other/prediction_result/balanced_dataset_csv_pred_result/coco_mAP.yaml`
* `other/prediction_result/balanced_dataset_csv_pred_result/imSitu_mAP.yaml`
* `other/prediction_result/fair_alm/celeba_alm_acc.yaml`
* `other/prediction_result/nifr/celeba_nifr_acc.yaml`

**From [RS3](#what-you-need-for-this-step-2)**: csv and json files under:

* `metric_calculation/s_aggregated_raw_prediction/result`
* `metric_calculation/c_aggregated_raw_prediction/result`
* `metric_calculation/i_aggregated_raw_prediction/result`
* `metric_calculation/a_aggregated_raw_prediction/result`
* `metric_calculation/n_aggregated_raw_prediction/result`



### Steps

The code for this step is under `other/bias_result_display`. You will need to follow these steps to get our results:

1. Copy the five model accuracy yaml files from RS2 to `other/bias_result_display/model_perf`

2. Copy the contents under `metric_calculation/s_aggregated_raw_prediction/result` (without parent folder) to `other/bias_result_display/raw_data/EffStrategies`

3. Copy the contents under `metric_calculation/c_aggregated_raw_prediction/result` (without parent folder) to `other/bias_result_display/raw_data/BalancedDatasets-COCO`

4. Copy the contents under `metric_calculation/i_aggregated_raw_prediction/result` (without parent folder) to `other/bias_result_display/raw_data/BalancedDatasets-imSitu`

5. Copy the contents under `metric_calculation/a_aggregated_raw_prediction/result` (without parent folder) to `other/bias_result_display/raw_data/FairALM-CelebA`

6. Copy the contents under `metric_calculation/n_aggregated_raw_prediction/result` (without parent folder) to `other/bias_result_display/raw_data/NIFR`

   After this step, the directory structure will be similar to this:

   ```bash
   other/bias_result_display/model_perf/
   ├── celeba_alm_acc.yaml
   ├── celeba_nifr_acc.yaml
   ├── cifar-10s.yaml
   ├── coco_mAP.yaml
   └── imSitu_mAP.yaml
   
   other/bias_result_display/raw_data/
   ├── BalancedDatasets-COCO
   │   ├── bias_amplification.csv
   │   ├── bias_amplification_raw.json
   │   ├── disparate_impact_factor.csv
   │   ├── disparate_impact_factor_raw.json
   │   ├── equality_of_odds_false_positive.csv
   │   ├── equality_of_odds_false_positive_raw.json
   │   ├── equality_of_odds_true_positive.csv
   │   ├── equality_of_odds_true_positive_raw.json
   │   ├── false_positive_subgroup_fairness.csv
   │   ├── false_positive_subgroup_fairness_raw.json
   │   ├── mean_difference_score.csv
   │   ├── mean_difference_score_raw.json
   │   ├── statistical_parity.csv
   │   └── statistical_parity_raw.json
   ├── BalancedDatasets-imSitu
   │   ├── bias_amplification.csv
   │   ├── bias_amplification_raw.json
   │   ├── disparate_impact_factor.csv
   │   ├── disparate_impact_factor_raw.json
   │   ├── equality_of_odds_false_positive.csv
   │   ├── equality_of_odds_false_positive_raw.json
   │   ├── equality_of_odds_true_positive.csv
   │   ├── equality_of_odds_true_positive_raw.json
   │   ├── false_positive_subgroup_fairness.csv
   │   ├── false_positive_subgroup_fairness_raw.json
   │   ├── mean_difference_score.csv
   │   ├── mean_difference_score_raw.json
   │   ├── statistical_parity.csv
   │   └── statistical_parity_raw.json
   ├── EffStrategies
   │   ├── bias_amplification.csv
   │   ├── bias_amplification_raw.json
   │   ├── disparate_impact_factor.csv
   │   ├── disparate_impact_factor_raw.json
   │   ├── equality_of_odds_false_positive.csv
   │   ├── equality_of_odds_false_positive_raw.json
   │   ├── equality_of_odds_true_positive.csv
   │   ├── equality_of_odds_true_positive_raw.json
   │   ├── false_positive_subgroup_fairness.csv
   │   ├── false_positive_subgroup_fairness_raw.json
   │   ├── mean_difference_score.csv
   │   ├── mean_difference_score_raw.json
   │   ├── statistical_parity.csv
   │   └── statistical_parity_raw.json
   ├── FairALM-CelebA
   │   ├── bias_amplification.csv
   │   ├── bias_amplification_raw.json
   │   ├── disparate_impact_factor.csv
   │   ├── disparate_impact_factor_raw.json
   │   ├── equality_of_odds_false_positive.csv
   │   ├── equality_of_odds_false_positive_raw.json
   │   ├── equality_of_odds_true_positive.csv
   │   ├── equality_of_odds_true_positive_raw.json
   │   ├── false_positive_subgroup_fairness.csv
   │   ├── false_positive_subgroup_fairness_raw.json
   │   ├── mean_difference_score.csv
   │   ├── mean_difference_score_raw.json
   │   ├── statistical_parity.csv
   │   └── statistical_parity_raw.json
   └── NIFR
       ├── bias_amplification.csv
       ├── bias_amplification_raw.json
       ├── disparate_impact_factor.csv
       ├── disparate_impact_factor_raw.json
       ├── equality_of_odds_false_positive.csv
       ├── equality_of_odds_false_positive_raw.json
       ├── equality_of_odds_true_positive.csv
       ├── equality_of_odds_true_positive_raw.json
       ├── false_positive_subgroup_fairness.csv
       ├── false_positive_subgroup_fairness_raw.json
       ├── mean_difference_score.csv
       ├── mean_difference_score_raw.json
       ├── statistical_parity.csv
       └── statistical_parity_raw.json
   ...
   ```

7. Execute the following scripts

   ```bash
   python3 0_dump_configs.py
   python3 1_0_load_raw_data.py
   python3 1_1_stat_result.py
   python3 2_0_baseline_vs_other.py
   python3 2_1_latex_tables.py
   python3 2_2_bias_mitigation_on_model_perf.py
   python3 2_3_mitigation_heatmap.py
   python3 2_4_migigation_cost.py
   python3 2_5_exp_overlap_count.py
   python3 2_6_result_for_submit.py
   ```

Now, you will have everything we need to write the paper.



## Result Files

Files in our https://github.com/lin-tan/fairness-variance repo can be find in these places:

* `raw_data.csv`: `other/bias_result_display/table/supp_material.csv`
* Everything under `raw_bias_numbers`: `other/bias_result_display/result_submit`



Figure 1 in the paper: `other/bias_result_display/figures/mean_bias_d.png`

Figure 2 in the paper: `other/bias_result_display/figures/variance_stat.png`

