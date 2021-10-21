import pickle
import pandas 
import json 
from pathlib import Path 

p = Path('/home/userQQQ/dlfairness/original_code/Balanced-Datasets-Are-Not-Enough/verb_classification/data', 'train.data')
with open(str(p), 'rb') as f:
    d = pickle.load(f)

train_set = [] # [{'idx': image_id, 'ground_truth': [objects], 'protected_label': gender}]
for image in d:
    image_dict = {'idx': image['image_name'], 'ground_truth': image['verb'], 'protected_label': image['gender']}
    train_set.append(image_dict)

with open('./imSitu_train.json', 'w') as f:
    json.dump(train_set, f)

    