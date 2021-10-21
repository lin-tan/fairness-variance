import argparse
import os
import json
import running_utils
import subprocess


def main():
    nas_dir = '/home/userHHH/Workspace/dlvariancetesting/result'
    working_dir = '/local2/userHHH/Workspace/dlvariancetesting/result'
    done_filename = 'train_done.csv'
    json_filename = 'variance_test_config.json'

    with open(json_filename) as config_file:
        configs = json.load(config_file)

    run_set, done_f = running_utils.setup_done_file(nas_dir, done_filename, ["run_type", "training_type", "dataset", "random_seed", "try"])
    done_f.close()

    for config in configs:
        # TODO: Do smart suggestion, for now hardcoded for testing
        no_try = config['no_tries']
        for iTry in range(no_try):
            # Create full config set
            main_conf = ('main_run', config['training_type'], config['dataset'], config['random_seed'], iTry, config['network'])
            eval_conf = ('eval_run', config['training_type'], config['dataset'], config['random_seed'], iTry, config['network'])

            # Eval command
            if 'eval_running_command' in config:
                eval_command = config['eval_running_command']
            else:
                eval_command = ''

            # Source folder
            source_dir = '%s/modified_training_files/%s_%s_%s_%s' \
                         % (config['shared_dir'], config['network'], config['training_type'], config['dataset'], config['random_seed'])

            # Check if this either config has been done before
            if eval_command != '':
                if main_conf in run_set and eval_conf not in run_set:
                    w_dir = working_dir + '/' + config['network'] + '_' + config['training_type'] + '_' + config['dataset'] + '_' + config['random_seed'] + '/' + str(iTry)
                    # print('cp ' + source_dir + '/DLVarLogNoLoop.csv' + ' ' + w_dir)
                    # print('cp ' + source_dir + '/' + config['eval_file'] + ' ' + w_dir)

                    print('cp -f ' + source_dir + '/DLVarLogNoLoop.csv' + ' ' + w_dir)
                    subprocess.call('cp -f ' + source_dir + '/DLVarLogNoLoop.csv' + ' ' + w_dir, shell=True)
                    subprocess.call('chmod 777 -R ' + w_dir + '/DLVarLogNoLoop.csv', shell=True)

                    print('cp -f ' + source_dir + '/pgd_config.json' + ' ' + w_dir)
                    subprocess.call('cp -f ' + source_dir + '/pgd_config.json' + ' ' + w_dir, shell=True)
                    subprocess.call('chmod 777 -R ' + w_dir + '/pgd_config.json', shell=True)

                    print('cp -f ' + source_dir + '/fgsm_config.json' + ' ' + w_dir)
                    subprocess.call('cp -f ' + source_dir + '/fgsm_config.json' + ' ' + w_dir, shell=True)
                    subprocess.call('chmod 777 -R ' + w_dir + '/fgsm_config.json', shell=True)

                    for eval_file in config['eval_file']:
                        print('cp -f ' + source_dir + '/' + eval_file+ ' ' + w_dir)
                        subprocess.call('cp -f ' + source_dir + '/' + eval_file + ' ' + w_dir, shell=True)
                        subprocess.call('chmod 777 -R ' + w_dir + '/' + eval_file, shell=True)

                    print('cp -f ' + source_dir + '/' + config['main_file'] + ' ' + w_dir)
                    subprocess.call('cp -f ' + source_dir + '/' + config['main_file'] + ' ' + w_dir, shell=True)
                    subprocess.call('chmod 777 -R ' + w_dir + '/' + config['main_file'], shell=True)

                    #subprocess.call('cp -f ' + source_dir + '/dl_logging_helper_no_loop.py' + ' ' + w_dir, shell=True)
                    #subprocess.call('chmod 777 -R ' + w_dir + '/dl_logging_helper_no_loop.py', shell=True)


if __name__ == "__main__":
    main()
