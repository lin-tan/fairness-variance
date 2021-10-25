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

## Table of Contents

1. RS1: [Executing the code for reproduced papers](<docs/RS1. Execute the code.md>)
    * [Technique "S"](<docs/RS1.1 Technique S.md>)
    * [Technique "C" and "I"](<docs/RS1.2 Technique C and I.md>)
    * [Technique "A"](<docs/RS1.3 Technique A.md>)
    * [Technique "N"](<docs/RS1.4 Technique N.md>)
2. RS2: [Collecting the prediction results](<docs/RS2. Collect prediction results.md>)
    * [Technique "S"](<docs/RS2.1 Technique S.md>)
    * [Technique "C" and "I"](<docs/RS2.2 Technique C and I.md>)
    * [Technique "A"](<docs/RS2.3 Technique A.md>)
    * [Technique "N"](<docs/RS2.4 Technique N.md>)
3. RS3: [Calculating the bias metrics](<docs/RS3. Bias metric calculation.md>)
4. RS4: [Generating the results](<docs/RS4. Generate final results.md>)
5. [Where to get the results](#result-files)

After the steps above, you will have everything we need to write the paper.

## Result Files

Files in our https://github.com/lin-tan/fairness-variance repo can be find in these places:

* `raw_data.csv`: `other/bias_result_display/table/supp_material.csv`
* Everything under `raw_bias_numbers`: `other/bias_result_display/result_submit`



Figure 1 in the paper: `other/bias_result_display/figures/mean_bias_d.png`

Figure 2 in the paper: `other/bias_result_display/figures/variance_stat.png`

