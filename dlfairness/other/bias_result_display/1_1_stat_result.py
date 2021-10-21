import yaml
import statistics
import csv

from pathlib import Path
from functools import partial
from scipy.stats import mannwhitneyu, levene

from abbrs import prefix_abbr, setting_abbr, metric_abbr
import decimal_utils
from utils import cohen_d


with open('./configs.yaml', 'r') as f:
    configs = yaml.full_load(f)
with open('./bias_metric.yaml', 'r') as f:
    bias_metric_list = yaml.full_load(f)

for paper, config in configs.items():
    print("Start:", paper)
    baseline, mitigation = config['settings']
    all_settings = baseline + mitigation
    baseline = baseline[0]

    with open(str(Path('./processed_data', paper, 'overall_raw.yaml')), 'r') as f:
        overall_raw = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): [value (each run)]}

    # Baseline vs. mitigation for all metrics
    for bias_metric in bias_metric_list:
        print("Start:", bias_metric)
        result = {}  # {bias_metric: {mean, norm_var, mean_p, norm_var_p, mean_d}}

        b_values = overall_raw[(baseline, bias_metric)]
        #if bias_metric == 'disparate_impact_factor':
        #    b_values = [1.0 - e for e in b_values] # Map for DI

        rl = partial(decimal_utils.round_list, digit=decimal_utils.GLOBAL_ROUND_DIGIT)
        b_mean = statistics.mean(rl(b_values))
        b_norm_var = statistics.variance(rl([e / float(b_mean) for e in b_values]))
        result[baseline] = {'mean': b_mean, 'norm_var': b_norm_var}
        for _mitigation in mitigation:
            m_values = overall_raw[(_mitigation, bias_metric)]
            #if bias_metric == 'disparate_impact_factor':
            #    m_values = [1.0 - e for e in m_values] # Map for DI

            m_mean = statistics.mean(rl(m_values))
            m_norm_var = statistics.variance(rl([e / float(m_mean) for e in m_values]))

            _, u_p = mannwhitneyu(b_values, m_values, alternative=('less' if b_mean < m_mean else 'greater'))
            _, l_p = levene([e / float(b_mean) for e in b_values], [e / float(m_mean) for e in m_values])
            d = cohen_d(b_values, m_values)

            result[_mitigation] = {'mean': m_mean, 'norm_var': m_norm_var, 'mean_p': u_p, 'norm_var_p': l_p, 'mean_d': d}

        p = Path('./stats/base_v_mitigation', paper)
        p.mkdir(exist_ok=True, parents=True)
        with open(str(Path(p, metric_abbr[bias_metric] + '.yaml')), 'w') as f:
            yaml.dump(result, f)

        with open(str(Path(p, metric_abbr[bias_metric] + '.csv')), 'w', newline='') as f:
            rf = partial(decimal_utils.round_significant_format, digit=decimal_utils.GLOBAL_ROUND_DIGIT, scientific=True)
            writer = csv.writer(f)
            writer.writerow(['Label', 'Mean', 'Mean_p', 'Norm_Var', 'Norm_var_p'])
            for setting in all_settings:
                row = [setting_abbr[paper][setting]]
                if setting == baseline:
                    row += [result[setting]['mean'], '', result[setting]['norm_var'], '']
                else:
                    row += [
                        result[setting]['mean'],
                        result[setting]['mean_p'],
                        result[setting]['norm_var'],
                        result[setting]['norm_var_p']
                    ]
                row = [rf(e) if not isinstance(e, str) else e for e in row]
                writer.writerow(row)

    # Improvement on all bias metrics
    for _mitigation in mitigation:
        result = {}  # {bias_metric: {stat: }}
        for bias_metric in bias_metric_list:
            rl = partial(decimal_utils.round_list, digit=decimal_utils.GLOBAL_ROUND_DIGIT)

            b_values = overall_raw[(baseline, bias_metric)]
            #if bias_metric == 'disparate_impact_factor':
            #    b_values = [1.0 - e for e in b_values] # Map for DI
            b_mean = statistics.mean(rl(b_values))
            b_norm_var = statistics.variance(rl([e / float(b_mean) for e in b_values]))
            m_values = overall_raw[(_mitigation, bias_metric)]
            #if bias_metric == 'disparate_impact_factor':
            #    m_values = [1.0 - e for e in m_values] # Map for DI
            m_mean = statistics.mean(rl(m_values))
            m_norm_var = statistics.variance(rl([e / float(m_mean) for e in m_values]))
            _, u_p = mannwhitneyu(b_values, m_values, alternative=('less' if b_mean < m_mean else 'greater'))
            _, l_p = levene([e / float(b_mean) for e in b_values], [e / float(m_mean) for e in m_values])

            result[metric_abbr[bias_metric]] = {
                'baseline_mean': b_mean,
                'mitigation_mean': m_mean,
                'mean_p': u_p,
                'baseline_norm_var': b_norm_var,
                'mitigation_norm_var': m_norm_var,
                'norm_var_p': l_p
            }

        p = Path('./stats/bias_metric_comparison', paper)
        p.mkdir(parents=True, exist_ok=True)
        with open(str(Path(p, setting_abbr[paper][_mitigation] + '.yaml')), 'w') as f:
            yaml.dump(result, f)
        with open(str(Path(p, setting_abbr[paper][_mitigation] + '.csv')), 'w', newline='') as f:
            rf = partial(decimal_utils.round_significant_format, digit=decimal_utils.GLOBAL_ROUND_DIGIT, scientific=True)
            writer = csv.writer(f)
            header = [
                'Label', 'Baseline', 'Mitigation', 'p_value', 'Baseline_Norm_Var', 'Mitigation_Norm_Var', 'p_value'
            ]
            for bias_metric in bias_metric_list:
                cur_stat = result[metric_abbr[bias_metric]]
                row = [
                    metric_abbr[bias_metric],
                    cur_stat['baseline_mean'],
                    cur_stat['mitigation_mean'],
                    cur_stat['mean_p'],
                    cur_stat['baseline_norm_var'],
                    cur_stat['mitigation_norm_var'],
                    cur_stat['norm_var_p']
                ]
                row = [rf(e) if not isinstance(e, str) else e for e in row]
                writer.writerow(row)
            