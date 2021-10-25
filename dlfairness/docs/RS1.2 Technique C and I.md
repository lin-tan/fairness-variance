# RS1: Technique "C" and "I"

**Acknowledgement: The code is adapted from https://github.com/uvavision/Balanced-Datasets-Are-Not-Enough**

The code is stored under `original_code/Balanced-Datasets-Are-Not-Enough/object_multilabel` (technique C) and `original_code/Balanced-Datasets-Are-Not-Enough/verb_classification` (technique I).

## Prepare the dataset

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

## Prepare the paths

Open `balanced_dataset_not_enough_coco.json` and fill in all the entries with keys `shared_dir`, `working_dir`, and `modified_target_dir` with the paths you choose.

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
