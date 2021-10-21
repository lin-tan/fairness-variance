import yaml
from pathlib import Path 
import glob

conf_list = ['CVPR', 'NIPS', 'ICML', 'AAAI', '*CCV', 'FAT']
for conf in conf_list:
    count = 0
    for yaml_file in glob.glob('./dump/' + conf + '/*.yaml'):
        with open(yaml_file, 'r') as f:
            paper_dict = yaml.load(f)
        count += len(paper_dict)
    print(conf, ':', count)
