import argparse
import pandas as pd
import numpy as np
import statistics as st
import json
import traceback
import scipy.stats as scst
import math


def cohens_d(c0, c1):
    sdev0 = st.stdev(c0)
    sdev1 = st.stdev(c1)

    if sdev0 == sdev1 == 0:
        c_d = 0
    else:
        c_d = (st.mean(c0) - st.mean(c1)) / (math.sqrt((sdev0 ** 2 + sdev1 ** 2) / 2))

    return c_d


def main():
    # read the parameter argument parsing
    parser = argparse.ArgumentParser(
        description='Comparing two sets of multiple runs')
    parser.add_argument('result_dir', help="the path of the result folder")
    parser.add_argument('run_configs_file', help="the json that stores the run configs")
    parser.add_argument('comparison_configs_file', help="the json that stores the pair of comparison")
    args = parser.parse_args()

    result_dir = args.result_dir
    run_configs_file = args.run_configs_file
    comparison_configs_file = args.comparison_configs_file

    with open(run_configs_file) as config_file:
        configs = json.load(config_file)

    data_dic = {}

    logfilename = 'DLVarLog.csv'
    analysis_out_name = 'comparison'

    analyze(analysis_out_name, comparison_configs_file, configs, data_dic, logfilename, result_dir)

    logfilename = 'DLVarLogNoLoop.csv'
    analysis_out_name = 'evel_comparison'

    analyze(analysis_out_name, comparison_configs_file, configs, data_dic, logfilename, result_dir)


def analyze(analysis_out_name, comparison_configs_file, configs, data_dic, logfilename, result_dir):
    for config in configs:
        print("Loading %s_%s_%s_%s" % (config['network'], config['training_type'], config['dataset'], config['random_seed']))

        try:
            no_tries = config['no_tries']

            dfs = []
            indexes = []
            for iTry in range(no_tries):
                df = pd.read_csv('%s/%s_%s_%s_%s/%d/%s' % (result_dir, config['network'], config['training_type'], config['dataset'], config['random_seed'], iTry, logfilename), skipinitialspace=True)

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

            data_dic["%s_%s_%s_%s" % (config['network'], config['training_type'], config['dataset'], config['random_seed'])] = data

        except Exception:
            print(traceback.format_exc())

    with open(comparison_configs_file) as config_file:
        comparing_pairs = json.load(config_file)

    for pair in comparing_pairs:

        approaches = pair['approaches']

        approach1 = approaches[0]
        key1 = "%s_%s_%s_%s" % (approach1['network'], approach1['training_type'], approach1['dataset'], approach1['random_seed'])
        if key1 not in data_dic:
            print("Missing %s" % key1)
            continue
        data1 = data_dic[key1]

        approach2 = approaches[1]
        key2 = "%s_%s_%s_%s" % (approach2['network'], approach2['training_type'], approach2['dataset'], approach2['random_seed'])
        if key2 not in data_dic:
            print("Missing %s" % key2)
            continue
        data2 = data_dic[key2]

        print("Analyzing %s of %s vs %s" % (logfilename, key1, key2))

        try:
            analysis_out = open('%s/%s_%s_%s.csv' % (result_dir, analysis_out_name, key1, key2), 'w')

            metrics = []

            for metric in data1.columns:
                if metric in ['Iter', 'try']:
                    continue
                metrics.append(metric)

            for metric in metrics:
                analysis_out.write(',' + metric + ',,')
            analysis_out.write('\n')

            analysis_out.write('Iter')
            for i in range(len(metrics)):
                analysis_out.write(',diff_of_mean,U_test_p-value,cohen_d')
            analysis_out.write('\n')

            for index in indexes:
                index_data1 = data1[data1['Iter'] == index]
                index_data2 = data2[data2['Iter'] == index]
                analysis_out.write('%d' % index)
                for metric in metrics:
                    values1 = index_data1[metric].values
                    values2 = index_data2[metric].values

                    diff_of_mean = st.mean(values1) - st.mean(values2)

                    if st.variance(values1) == 0 and st.variance(values2) == 0 and values1[0] == values2[0]:
                        p_value = np.nan
                    else:
                        _, p_value = scst.mannwhitneyu(values1, values2)

                    cd = cohens_d(values1, values2)

                    analysis_out.write(',%.4f,%.4f,%.4f' % (diff_of_mean, p_value, cd))

                analysis_out.write('\n')

            analysis_out.close()
        except Exception:
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
