import csv
import yaml

papers = {}
with open('./dump/ECCV/ECCV2016.csv', 'r', newline='', encoding='utf-8', errors='ignore') as f:
    r = csv.reader(f)
    for (title, link) in r:
        papers[title] = (link, 2016)

with open('./dump/ECCV/2.yaml', 'w') as f:
    yaml.safe_dump(papers, f)
