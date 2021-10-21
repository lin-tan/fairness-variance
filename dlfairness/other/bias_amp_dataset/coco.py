import pickle
import pandas 
import json 
from pathlib import Path 

p = Path('/home/userQQQ/dlfairness/original_code/Balanced-Datasets-Are-Not-Enough/object_multilabel/data', 'train.data')
with open(str(p), 'rb') as f:
    d = pickle.load(f)

train_set = [] # [{'idx': image_id, 'ground_truth': [objects], 'protected_label': gender}]
for image in d:
    image_dict = {'idx': image['image_id'], 'ground_truth': image['objects']}
    if image['gender'][0] == 1:
        image_dict['protected_label'] = 0
    else:
        image_dict['protected_label'] = 1

    train_set.append(image_dict)

with open('./coco_train.json', 'w') as f:
    json.dump(train_set, f)

    