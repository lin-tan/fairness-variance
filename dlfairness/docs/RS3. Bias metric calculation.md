# RS3. Bias Metric Calculation

## What you need for this step

**From RS2**: Contents under all the `OUTPUT_DIR`s in RS2, including the parent folder `OUTPUT_DIR`



## Overview

The code for bias metric calculation is self-contained and is located under `other/metric_calculation`. You will need the content of all the `OUTPUT_DIR`s in the previous section (RS2).

Details of bias metric calculation code can be found [here](../other/metric_calculation/README.md).



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



Assume the `OUTPUT_DIR`s follow the example in the previous section ([RS2](#rs2-collecting-prediction-results)), five folders will be created under `other/metric_calculation`:

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