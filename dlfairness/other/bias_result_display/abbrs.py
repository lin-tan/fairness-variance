metric_abbr = {
    'statistical_parity': 'SPSF',
    'false_positive_subgroup_fairness': 'FPSF',
    'disparate_impact_factor': 'DI',
    'mean_difference_score': 'DP',
    'equality_of_odds_true_positive': 'EOTP',
    'equality_of_odds_false_positive': 'EOFP',
    'bias_amplification': 'BA'
}

metric_display = {
    'statistical_parity': 'Statistical Parity',
    'false_positive_subgroup_fairness': 'False Positive Subgroup Fairness',
    'disparate_impact_factor': 'Disparate Impact',
    'mean_difference_score': 'Demographic Parity',
    'equality_of_odds_true_positive': 'Equalized Odds - TP',
    'equality_of_odds_false_positive': 'Equalized Odds - FP',
    'bias_amplification': 'Bias Amplification'
}

prefix_abbr = {
    'BalancedDatasets-COCO': 'C', 'BalancedDatasets-imSitu': 'I', 'EffStrategies': 'S', 'FairALM-CelebA': 'A', 'NIFR': 'N'
}

setting_abbr = {
    'BalancedDatasets-COCO': {
        'threshold_0.5/no_gender/': 'C-Base',
        'threshold_0.5/ratio-1/': 'C-R1',
        'threshold_0.5/ratio-2/': 'C-R2',
        'threshold_0.5/ratio-3/': 'C-R3',
        'threshold_0.5/adv-conv4/': 'C-A4',
        'threshold_0.5/adv-conv5/': 'C-A5'
    },
    'BalancedDatasets-imSitu': {
        'no_gender/': 'I-Base',
        'ratio-1/': 'I-R1',
        'ratio-2/': 'I-R2',
        'ratio-3/': 'I-R3',
        'adv-conv4/': 'I-A4',
        'adv-conv5/': 'I-A5'
    },
    'EffStrategies': {
        'baseline/': 'S-Base',
        'sampling/': 'S-RS',
        'uniconf-adv/': 'S-UC',
        'gradproj-adv/': 'S-GR',
        'domain-discriminative/Sum-of-prob-no-prior-shift/': 'S-DD1',
        'domain-discriminative/Max-of-prob-prior-shift/': 'S-DD2',
        'domain-discriminative/Sum-of-prob-prior-shift/': 'S-DD3',
        'domain-discriminative/rba/': 'S-DD4',
        'domain-independent/Conditional/': 'S-DI1',
        'domain-independent/Sum/': 'S-DI2'
    },
    "FairALM-CelebA": {
        'no-constraints/': 'A-Base', 'l2-penalty/': 'A-L2', 'fair-alm/': 'A-ALM'
    },
    "NIFR": {
        'baseline/': 'N-Base', 'inn/': 'N-Flow'
    }
}

setting_order = ['EffStrategies', 'BalancedDatasets-COCO', 'BalancedDatasets-imSitu', 'FairALM-CelebA', 'NIFR']
