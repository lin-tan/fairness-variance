# RS4. Generate Final Results

## What you need for this step

**From [RS2](#rs2-collecting-prediction-results)**: yaml files containing model accuracy:

* `other/prediction_result/dbm_csv_pred_result/cifar-10s.yaml`
* `other/prediction_result/balanced_dataset_csv_pred_result/coco_mAP.yaml`
* `other/prediction_result/balanced_dataset_csv_pred_result/imSitu_mAP.yaml`
* `other/prediction_result/fair_alm/celeba_alm_acc.yaml`
* `other/prediction_result/nifr/celeba_nifr_acc.yaml`

**From [RS3](#rs3-bias-metric-calculation)**: csv and json files under:

* `metric_calculation/s_aggregated_raw_prediction/result`
* `metric_calculation/c_aggregated_raw_prediction/result`
* `metric_calculation/i_aggregated_raw_prediction/result`
* `metric_calculation/a_aggregated_raw_prediction/result`
* `metric_calculation/n_aggregated_raw_prediction/result`



## Steps

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
