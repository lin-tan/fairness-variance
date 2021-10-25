# RS1: Technique "A"

**Acknowledgement: The code is adapted from https://github.com/lokhande-vishnu/FairALM**

The code is stored under `original_code/FairALM/Experiments-CelebA`.

## Prepare the dataset

```bash
# CelebA
cd original_code/FairALM/Experiments-CelebA/data
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/img_align_celeba.zip
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/list_attr_celeba.txt
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/list_eval_partition.txt
unzip img_align_celeba.zip
```

## Prepare the paths

Open `fair_alm.json` and fill in all the entries with keys `shared_dir`, `working_dir`, and `modified_target_dir` with the paths you choose.

Open `0_1_1_training_code_modify_fairness.sh`, modify `NAS_DIR` on Line 2 and `json_filename` on Line 4.

Open `1_1_1_setup_server_runs_fairness.sh`, modify `NAS_DIR` on Line 2 and `json_filename` on Line 5.

Open `1_2_1_multiple_server_runs_deepgpu3_fairness.sh`, modify `RESULT_DIR` on Line 2 and `NAS_DIR` on Line 3.

## Run the scripts

```bash
bash 0_1_1_training_code_modify_fairness.sh
bash 1_1_1_setup_server_runs_fairness.sh
bash 1_2_1_multiple_server_runs_deepgpu3_fairness.sh
```

## Results

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