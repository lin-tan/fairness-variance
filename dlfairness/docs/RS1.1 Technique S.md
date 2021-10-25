# RS1: Technique "S"

**Acknowledgement: The code is adapted from https://github.com/princetonvisualai/DomainBiasMitigation**

The code is stored under `original_code/DomainBiasMitigation`.

## Prepare the dataset

```bash
cd original_code/DomainBiasMitigation
bash download.sh
python3 preprocess_data.py
```

## Prepare the paths

Open `fairness_variance_test_config.json` and fill in all the entries with keys `shared_dir` and `working_dir` with the paths you choose.

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