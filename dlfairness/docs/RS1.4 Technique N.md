# RS1: Technique "N"

**Acknowledgement: The code is adapted from https://github.com/predictive-analytics-lab/nifr**

The code is stored under `original_code/nifr`.

## Prepare the dataset

```bash
# CelebA
cd original_code/nifr/data/celeba
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/img_align_celeba.zip
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/list_attr_celeba.txt
wget https://github.com/lin-tan/fairness-variance/releases/download/dataset/list_eval_partition.txt
unzip img_align_celeba.zip
```

## Prepare the paths

Open `nifr.json` and fill in all the entries with keys `shared_dir`, `working_dir`, and `modified_target_dir` with the paths you choose.

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
├── NIFR_baseline_CelebA_fixed1
├── NIFR_inn_CelebA_fixed1
└── train_per_epoch_21_01_23_18_37_42.log
```