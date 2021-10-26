# RS2. Collecting Prediction Results

In this step, you will collect the predicted task label on the testset with each of the trained model collected from RS1. Generally speaking, each technique will be split into two parts:
* Modify the path in the script
* Collect result

You can read [RS2.1](<RS2.1 Technique S.md>) for Technique "S", [RS2.2](<RS2.2 Technique C and I.md>) for Technique "C" and "I", [RS2.3](<RS2.3 Technique A.md>), and [RS2.4](<RS2.4 Technique N.md>) for Technique "N" in the paper. Each of the file contains the two parts above.

## What you need for this step

**From RS1**: `shared_dir`, `working_dir`, `modified_target_dir`, `NAS_DIR`, and `RESULT_DIR` for all four reproduced papers

| Technique | shared_dir  | working_dir  | modified_target_dir | NAS_DIR              | RESULT_DIR            |
| --------- | ----------- | ------------ | ------------------- | -------------------- | --------------------- |
| S         | /dlfairness | /working_dir | N/A                 | /dlfairness/result_s | /working_dir/result_s |
| C, I      | /dlfairness | /working_dir | /modified_dir       | /dlfairness/result_c | /working_dir/result_c |
| A         | /dlfairness | /working_dir | /modified_dir       | /dlfairness/result_a | /working_dir/result_a |
| N         | /dlfairness | /working_dir | /modified_dir       | /dlfairness/result_n | /working_dir/result_n |


**Define** `OUTPUT_DIR` for each set of techniques with the same initial

In addition of the five paths mentioned above (`shared_dir`, `working_dir`, `modified_target_dir`, `NAS_DIR`, and `RESULT_DIR`), we will need another path `OUTPUT_DIR` to store all the prediction results.

| Technique | OUTPUT_DIR                                        |
| --------- | ------------------------------------------------- |
| S         | /working_dir/result_s/s_aggregated_raw_prediction |
| C         | /working_dir/result_c/c_aggregated_raw_prediction |
| I         | /working_dir/result_c/i_aggregated_raw_prediction |
| A         | /working_dir/result_a/a_aggregated_raw_prediction |
| N         | /working_dir/result_n/n_aggregated_raw_prediction |
