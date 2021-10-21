import argparse
import pandas as pd
import numpy as np
import statistics as st
import math
import json
import traceback
import scipy.stats


def main():
    # read the parameter argument parsing
    parser = argparse.ArgumentParser(
        description='Analyze the variance of multiple runs')
    parser.add_argument('result_dir', help="the path of the result folder")
    parser.add_argument('json_filename', help="the json filename")
    args = parser.parse_args()

    result_dir = args.result_dir
    json_filename = args.json_filename

    with open(json_filename) as config_file:
        configs = json.load(config_file)

    for config in configs:
        print("Performing analysis of %s_%s_%s_%s" % (config['network'], config['training_type'], config['dataset'], config['random_seed']))

        out_name = 'variance_analysis'
        log_name = 'DLVarLog.csv'
        analyze_result(config, log_name, out_name, result_dir)

        if 'eval_running_command' in config:
            eval_out_name = 'eval_variance_analysis'
            eval_log_name = 'DLVarLogNoLoop.csv'
            analyze_result(config, eval_log_name, eval_out_name, result_dir)


def sdev_confidence_interval(data, confidence=0.9):
    a = 1.0 * np.array(data)
    n = len(a)
    sdev = st.stdev(a)
    t = sdev * math.sqrt((n-1)/scipy.stats.distributions.chi2.ppf((1 - confidence) / 2., n-1))
    b = sdev * math.sqrt((n-1)/scipy.stats.distributions.chi2.ppf(1 - (1 - confidence) / 2., n-1))
    return sdev, b, t


def analyze_result(config, log_name, out_name, result_dir):
    try:

        no_tries = config['no_tries']

        analysis_out = open('%s/%s_%s_%s_%s_%s.csv' % (result_dir, out_name, config['network'], config['training_type'], config['dataset'], config['random_seed']), 'w')

        dfs = []
        indexes = []
        for iTry in range(no_tries):
            df = pd.read_csv('%s/%s_%s_%s_%s/%d/%s' % (result_dir, config['network'], config['training_type'], config['dataset'], config['random_seed'], iTry, log_name), skipinitialspace=True)

            iters = df['Iter'].values
            if st.mean(iters) == iters[0]:
                if len(indexes) == 0 or len(indexes) > len(iters):
                    indexes = range(len(iters))
                df['Iter'] = indexes
            else:
                if len(indexes) == 0 or len(indexes) > len(df['Iter'].values):
                    indexes = df['Iter'].values

            iTries = [iTry] * len(df['Iter'].values)
            df['try'] = iTries
            dfs.append(df)

        data = pd.concat(dfs)

        index = indexes[-1]
        index_data = data[data['Iter'] == index]

        metrics_dic = {}

        for metric in index_data.columns:
            if metric in ['Iter', 'try']:
                continue

            values = index_data[metric]

            std_dev = st.stdev(values)
            mean = st.mean(values)

            if np.isnan(values).any():
                relative_std_dev = 0
            elif mean == 0:
                relative_std_dev = std_dev
            else:
                relative_std_dev = std_dev / mean

            metrics_dic[metric] = relative_std_dev

        metrics = sorted(metrics_dic, key=metrics_dic.get, reverse=True)

        for metric in metrics:
            analysis_out.write(',' + metric + ',,,,,,')
        analysis_out.write('\n')

        analysis_out.write('Iter')
        for i in range(len(metrics)):
            analysis_out.write(',max_diff,max,min,SDev,SDev_t,SDev_b,mean')
        analysis_out.write('\n')

        for index in indexes:
            index_data = data[data['Iter'] == index]
            analysis_out.write('%d' % index)
            for metric in metrics:
                values = index_data[metric]

                max_v = max(values)
                min_v = min(values)
                max_diff = max_v - min_v
                std_dev, t, b = sdev_confidence_interval(values)
                mean = st.mean(values)

                analysis_out.write(',%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (max_diff, max_v, min_v, std_dev, t, b, mean))

            analysis_out.write('\n')

        analysis_out.close()
    except Exception:
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
