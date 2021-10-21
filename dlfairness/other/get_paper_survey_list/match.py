import yaml
import argparse
import glob
import re
from pathlib import Path
import csv
import nltk


keywords = [
    'fair',
    'bias',
    'discriminate',
    'aware',
    'balance',
    'disparity',
    'opportunity',
    'audit',
    'gender',
    'stereotype',
    'race',
    'amplification',
    'ethic'
]

#keywords = ['ethic']

def normalize_word(s):
    stemmer = nltk.stem.SnowballStemmer('english')
    s = re.sub('\W+', '', s)
    s = s.lower()
    s = stemmer.stem(s)
    return s

def list_contain(tlist, word):
    for w in tlist:
        if w.find(word) != -1:
        #if w == word:
            return True
    return False

def title_match(title, keywords):
    title_list = [normalize_word(e.strip()) for e in title.split(' ')]
    for keyword in keywords:
        if list_contain(title_list, normalize_word(keyword)):
            tlist = [e.encode('ascii', errors='ignore').decode('utf-8') for e in title_list]
            print(keyword, ':', tlist)
            return True 
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    args = parser.parse_args()

    conf = args.conf
    matches = {} # {conf: {year, {title: link}}}

    if conf not in matches:
        matches[conf] = {}
    
    for yaml_file in glob.glob('./dump/' + conf + '/*.yaml'):
        print(yaml_file)
        with open(yaml_file, 'r') as f:
            paper_dict = yaml.safe_load(f)
    
        for title, (paper_link, year) in paper_dict.items(): 
            if title_match(title, keywords):
                if year not in matches[conf]:
                    matches[conf][year] = {}
                matches[conf][year][title] = paper_link

    
        for conf, conf_dict in matches.items():
            p = Path('./csv')
            p.mkdir(exist_ok=True, parents=True)
            with open(str(Path(p, conf + '.csv')), 'w', newline='', encoding='utf-8') as f:
            #with open(str(Path(p, conf + '.csv')), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Conf', 'Title', 'Link'])
                year_sorted = sorted(list(conf_dict.keys()), reverse=True)
                for year in year_sorted:
                    for title, paper_link in conf_dict[year].items():
                        writer.writerow([conf + str(year), title, paper_link])

if __name__ == '__main__':
    main()