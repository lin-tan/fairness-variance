import shutil, random, os
from pathlib import Path
from PIL import Image

dir_path = Path('./raw_data')
dest_path = Path('./tuberculosis-data-processed/')

fn_list = list(dir_path.iterdir())
test_set = random.sample(fn_list, int(len(fn_list) * 0.25)) # Sample 25%

train_dir = Path(dest_path, 'train')
test_dir = Path(dest_path, 'test')
train_dir.mkdir(exist_ok=True, parents=True)
test_dir.mkdir(exist_ok=True, parents=True)

for fn in fn_list:
    image = Image.open(fn).resize((128, 128)).convert('L')
    if fn in test_set:
        save_path = Path(test_dir, fn.name)
    else:
        save_path = Path(train_dir, fn.name)

    image.save(save_path)