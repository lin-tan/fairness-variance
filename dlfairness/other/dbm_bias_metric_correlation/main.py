import scipy.stats
import json
from pathlib import Path 
import csv

bias_metric_list = [
    'statistical_parity',
    'false_positive_subgroup_fairness',
    'disparate_impact_factor',
    'mean_difference_score',
    'bias_amplification_2'
]

exp_list = [
    'baseline/',
    'sampling/',
    'uniconf-adv/',
    'gradproj-adv/',
    'domain-discriminative/Sum-of-prob-no-prior-shift/',
    'domain-discriminative/Max-of-prob-prior-shift/',
    'domain-discriminative/Sum-of-prob-prior-shift/',
    'domain-discriminative/rba/',
    'domain-independent/Conditional/',
    'domain-independent/Sum/'
]

def main():
    # Load bias metrics
    raw_bias_score = {} # {metric: {exp: [overall_score]}}
    for bias_metric in bias_metric_list:
        p = Path('./data', bias_metric + '_raw.json')
        with open(str(p), 'r') as f:
            d = json.load(f)
        raw_bias_score[bias_metric] = {}

        for exp in exp_list:
            if bias_metric == 'bias_amplification_2':
                raw_bias_score[bias_metric][exp] = d[exp]
                continue 

            raw_bias_score[bias_metric][exp] = []
            for score_list in d[exp]:
                raw_bias_score[bias_metric][exp].append(score_list[-1]) # Get the overall score only
            
            #print(raw_bias_score[bias_metric][exp])

    # Write the correlation
    with open('./dbm_correlation.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for exp in exp_list:
            w.writerow([exp])
            w.writerow([''] + bias_metric_list)
            for metric_1 in bias_metric_list:
                row = [metric_1]
                for metric_2 in bias_metric_list:
                    s1 = raw_bias_score[metric_1][exp]
                    s2 = raw_bias_score[metric_2][exp]
                    pearsonr, _ = scipy.stats.pearsonr(s1, s2)
                    row.append(pearsonr)
                w.writerow(row)
            w.writerow([])
    
    # Aggregate all the stats
    bias_stat = {} # {Exp: {Metric: [max_diff, max, min, stdev, mean, rel_maxdiff(%)]}}
    for bias_metric in bias_metric_list:
        p = Path('./data', bias_metric + '.csv')
        with open(str(p), 'r', newline='') as f:
            r = csv.reader(f)
            for row in r:
                if 'rel_maxdiff(%)' in row:
                    continue # Skip header
                
                exp = row[0][28:] # Strip 
                if bias_metric == 'bias_amplification_2':
                    bias_stat.setdefault(exp, {})
                    bias_stat[exp][bias_metric] = row[1:]
                else:
                    if row[1] == 'overall': # We only use overall for this
                        bias_stat.setdefault(exp, {})
                        bias_stat[exp][bias_metric] = row[2:]
    
    with open('./aggregated_overall_stat.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for exp in exp_list:
            #print(bias_stat[exp].keys())
            w.writerow([exp])
            w.writerow(['', 'max_diff', 'max', 'min', 'stdev', 'mean', 'rel_maxdiff'])
            for bias_metric in bias_metric_list:
                w.writerow([bias_metric] + bias_stat[exp][bias_metric])
            w.writerow([])

if __name__ == '__main__':
    main()