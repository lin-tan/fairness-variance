import traceback
import training_code_metric_processing
import os
import subprocess
import argparse
import json
import sys
from timeit import default_timer as timer
import running_utils


def main():
    # read the parameter argument parsing
    parser = argparse.ArgumentParser(
        description='Modify the training code')
    parser.add_argument('nas_dir', help="the path of the shared folder")
    parser.add_argument('done_filename', help="the done filename")
    parser.add_argument('json_filename', help="the json filename")

    args = parser.parse_args()

    nas_dir = args.nas_dir
    done_filename = args.done_filename
    json_filename = args.json_filename

    with open(json_filename) as config_file:
        configs = json.load(config_file)

    run_set, done_f = running_utils.setup_done_file(nas_dir, done_filename, ["training_type", "dataset", "random_seed"])

    for config in configs:
        conf = (config['training_type'], config['dataset'], config['random_seed'], config['network'])

        # Check if this config has been done before
        if conf in run_set:
            continue

        source_dir = '%s/%s' % (config['shared_dir'], config['source_dir'])
        if "modified_target_dir" in config:
            modified_dir = '%s/%s/%s_%s_%s_%s' % (config['modified_target_dir'], "modified_training_files", config['network'], config['training_type'], config['dataset'], config['random_seed'])
        else:
            modified_dir = '%s/%s/%s_%s_%s_%s' % (config['shared_dir'], "modified_training_files", config['network'], config['training_type'], config['dataset'], config['random_seed'])

        filename = '%s/%s' % (source_dir, config['main_file'])
        outfilename = '%s/%s' % (modified_dir, config['main_file'])
        logfilename = '%s/DLVarLog.csv' % modified_dir

        try:
            #subprocess.call('rm -rf %s' % modified_dir, shell=True)
            #os.makedirs(modified_dir, exist_ok=True)

            #subprocess.call('cp -a %s/. %s/' % (source_dir, modified_dir), shell=True)
            subprocess.call('rsync -a %s/. %s/' % (source_dir, modified_dir), shell=True)

            begin = timer()

            '''
            # modify the training code

            print("Begin loops dectetion")
            sys.stdout.flush()

            # loops = training_code_metric_processing.extract_metric(filename)

            print("End loops dectetion and begin modification")
            sys.stdout.flush()

            subprocess.call('python 0_1_0_single_file_modify.py ' +
                            filename + ' ' + outfilename + ' ' + logfilename + ' loops',
                            stdout=sys.stdout, stderr=sys.stderr, shell=True)

            # training_code_metric_processing.modify_file(filename, None, outfilename, logfilename)

            print("End modification")
            sys.stdout.flush()

            print('Done: ' + outfilename)

            # modify evaluation code

            # TODO: check config['eval_file'] is a list or not
            for evalfile in config['eval_file']:
                evalfilename = '%s/%s' % (source_dir, evalfile)
                evaloutfilename = '%s/%s' % (modified_dir, evalfile)
                evallogfilename = '%s/DLVarLogNoLoop.csv' % modified_dir

                print("Begin metrics dectetion")
                sys.stdout.flush()

                # funcs = training_code_metric_processing.extract_metric_without_main_loop(filename)

                print("End metrics dectetion and begin modification")
                sys.stdout.flush()

                subprocess.call('python 0_1_0_single_file_modify.py ' +
                                evalfilename + ' ' + evaloutfilename + ' ' + evallogfilename + ' metrics',
                                stdout=sys.stdout, stderr=sys.stderr, shell=True)
                # training_code_metric_processing.modify_file_with_metrics(evalfilename, None, evaloutfilename, evallogfilename)

                print("End modification")
                sys.stdout.flush()

                print('Done: ' + evaloutfilename)
            '''

            subprocess.call('cp ' + 'dl_logging_helper.py ' + os.path.dirname(outfilename), shell=True)

            #subprocess.call('cp ' + 'dl_logging_helper_no_loop.py ' + os.path.dirname(evaloutfilename), shell=True)

            subprocess.call('cp ' + '1_0_execute_single_run.sh ' + modified_dir, shell=True)
            subprocess.call('cp ' + '1_0_1_execute_single_run_no_conda.sh ' + modified_dir, shell=True)

            end = timer()

            done_f.write('%s,%s,%s,%s,%.5f\n' % (config['network'], config['training_type'], config['dataset'], config['random_seed'], (end - begin)))
            done_f.flush()

            # del loops
        except Exception:
            print(filename)
            print(traceback.format_exc())

    done_f.close()


if __name__ == "__main__":
    main()
