import yaml

bias_metric_list = [
    'statistical_parity',
    'false_positive_subgroup_fairness',
    'disparate_impact_factor',
    'mean_difference_score', # Demographic parity
    'equality_of_odds_true_positive',
    'equality_of_odds_false_positive',
    'bias_amplification'
]

configs = {
    # CIFAR-10S, Color/Gray
    "EffStrategies": {
        'settings': (
            ['baseline/'],
            [
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
        ),
        'class': 10,
        'performance_file': 'cifar-10s.yaml'
    },
    "BalancedDatasets-COCO": {
        'settings': (
            ['threshold_0.5/no_gender/'],
            [
                'threshold_0.5/ratio-1/',
                'threshold_0.5/ratio-2/',
                'threshold_0.5/ratio-3/',
                'threshold_0.5/adv-conv4/',
                'threshold_0.5/adv-conv5/'
            ]
        ),
        'class': 79,
        'performance_file': 'coco_mAP.yaml'
    },
    "BalancedDatasets-imSitu": {
        'settings': (
            ['no_gender/'],
            [
                'ratio-1/',
                'ratio-2/',
                'ratio-3/',
                'adv-conv4/',
                'adv-conv5/'
            ]
        ),
        'class': 211,
        'performance_file': 'imSitu_mAP.yaml'
    },
    "FairALM-CelebA": {
        'settings': (
            ['no-constraints/'],
            [
                'l2-penalty/',
                'fair-alm/'
            ]
        ),
        'class': 2,
        'performance_file': 'celeba_alm_acc.yaml'
    },
    "NIFR": {
        'settings': (
            ['baseline/'],
            ['inn/']
        ),
        'class': 2,
        'performance_file': 'celeba_nifr_acc.yaml'
    }
}

with open('./bias_metric.yaml', 'w') as f:
    yaml.dump(bias_metric_list, f)

with open('./configs.yaml', 'w') as f:
    yaml.dump(configs, f)
