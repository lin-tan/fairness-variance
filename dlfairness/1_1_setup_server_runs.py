import argparse
import os
import json
import running_utils


def main():
    # read the parameter argument parsing
    parser = argparse.ArgumentParser(
        description='Create a list of experiments')
    parser.add_argument('nas_dir', help="the path of the shared nas folder")
    parser.add_argument('done_filename', help="the done filename")
    parser.add_argument('queue_filename', help="the queue filename")
    parser.add_argument('json_filename', help="the json filename")

    args = parser.parse_args()

    nas_dir = args.nas_dir
    done_filename = args.done_filename
    queue_filename = args.queue_filename
    json_filename = args.json_filename

    with open(json_filename) as config_file:
        configs = json.load(config_file)

    run_set, done_f = running_utils.setup_done_file(nas_dir, done_filename, ["run_type", "training_type", "dataset", "random_seed", "try"])
    done_f.close()

    gpu_runs = []

    print(run_set)

    for config in configs:
        # TODO: Do smart suggestion, for now hardcoded for testing
        no_try = config['no_tries']
        for iTry in range(no_try):
            # Create full config set
            main_conf = ('main_run', config['training_type'], config['dataset'], config['random_seed'], iTry, config['network'])
            eval_conf = ('eval_run', config['training_type'], config['dataset'], config['random_seed'], iTry, config['network'])

            # Main command
            if isinstance(config['running_command'], list):
                command = ';'.join(config['running_command'])
            else:
                command = config['running_command']

            # Eval command
            if 'eval_running_command' in config:
                if isinstance(config['eval_running_command'], list):
                    eval_command = ';'.join(config['eval_running_command'])
                else:
                    eval_command = config['eval_running_command']
            else:
                eval_command = ''

            # Log file
            log_filename = '1_%s_%s_%s_%s_%d.log' % (config['network'], config['training_type'], config['dataset'], config['random_seed'], iTry)

            # Conda environment
            env = config['conda_env']

            # Docker container and images
            docker_image = config['docker_env']
            docker_container = docker_image.replace(':', '_').upper() + '_GPU_%d'

            # Source folder
            if "modified_target_dir" in config:
                source_dir = '%s/modified_training_files/%s_%s_%s_%s' % (config['modified_target_dir'], config['network'], config['training_type'], config['dataset'], config['random_seed'])
            else:
                source_dir = '%s/modified_training_files/%s_%s_%s_%s' % (config['shared_dir'], config['network'], config['training_type'], config['dataset'], config['random_seed'])

            # Docker environment
            if env == '##_NO_CONDA_##':
                docker_command = 'docker run ' + '-w /working ' + '-v /dev/shm:/dev/shm '
            else:
                docker_command = 'docker run ' + \
                                '-w /working ' + \
                                '-v /usr/local/anaconda3:/usr/local/anaconda3 ' + \
                                '-v /home/userHHH/.conda:/home/userHHH/.conda ' + \
                                '-v /local/userHHH/env:/env'

            # Check if this either config has been done before
            if main_conf in run_set:
                if eval_command == '':
                    continue

                if eval_conf in run_set:
                    continue

            gpu_runs.append(
                (config['network'], config['training_type'], config['dataset'], config['random_seed'], iTry,
                 source_dir, docker_command, docker_container, docker_image, env, command, eval_command, log_filename))

    queue_f = open(nas_dir + '/' + queue_filename, 'w')
    for run in gpu_runs:
        (network, training_type, dataset, random_seed, iTry,
         source_dir, docker_command, docker_container, docker_image, env, command, eval_command, log_filename) = run
        queue_f.write(network + '\n')
        queue_f.write(training_type + '\n')
        queue_f.write(dataset + '\n')
        queue_f.write(random_seed + '\n')
        queue_f.write(str(iTry) + '\n')
        queue_f.write(source_dir + '\n')
        queue_f.write(docker_command + '\n')
        queue_f.write(docker_container + '\n')
        queue_f.write(docker_image + '\n')
        queue_f.write(env + '\n')
        queue_f.write(command + '\n')
        queue_f.write(eval_command + '\n')
        queue_f.write(log_filename + '\n')
    queue_f.close()


if __name__ == "__main__":
    main()
