# Are My Deep Learning Systems Fair? An Empirical Study of Fixed-Seed Training

This is the artifact repository for the NeurIPS 2021 paper: Are My Deep Learning Systems Fair? An Empirical Study of Fixed-Seed Training.

## Artifacts list

| URL                                                          | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [raw_data.csv](raw_data.csv)                                 | This spreadsheet is the overall and per-class fairness variance results for 189 experiments (27 techniques and seven bias metrics). |
| [stat_tests.csv](raw_bias_numbers/stat_tests.csv)            | This spreadsheet contains the statistical test results for 154 bias mitigation experiments (22 mitigation techinques and seven bias metrics), including U-test, Levene's test, and Cohen's d number. |
| [overall](raw_bias_numbers/overall)                          | Naming convention: ```./raw_bias_numbers/overall/<Technique>.yaml```<br>This folder contains the raw overall bias numbers for each technique from 16 FIT runs. <br>Each yaml file is a dictionary. The key of the dictionary is the bias metric. The value of the dictionary is a list of 16 bias values for each run. |
| [per-class](raw_bias_numbers/per_class)                      | Naming convention: ```./raw_bias_numbers/per_class/<Technique>/run_<idx>.yaml```<br>This folder contains the raw per-class bias numbers for each technique from 16 FIT runs. <br>Each yaml file is a dictionary, and contains the bias value for all the classes in one run. The key of the dictionary is the bias metric.  The value of the dictionary is a list of bias values for each class. |
| [Raw prediction results](https://github.com/lin-tan/fairness-variance/releases/tag/prediction) | This release contains the raw prediction results of 27 techniques in the paper, while each technique has the result for 16 FIT runs. |
| [Docker image](https://github.com/lin-tan/fairness-variance/releases/tag/docker_image) | This release contains the docker image to reproduce our study. |
| [Weights](https://github.com/lin-tan/fairness-variance/releases/tag/weight) | This release contains the weights for all the trained models. |
| [Dataset](https://github.com/lin-tan/fairness-variance/releases/tag/dataset) | This release contains a backup of all the datasets used in our study. |
| [Replication package](dlfairness)                            | This folder contains the code and scripts to reproduce our study. |
| [Code README](dlfairness/README.md)                          | This file is the guide to use our replication package.       |
| [Bias metric calculation](dlfairness/other/metric_calculation/README.md) | This file is the details about the bias metric calculation part of the replication package. |

