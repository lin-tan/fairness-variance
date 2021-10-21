import argparse
import subprocess
from multiprocessing import Lock, Queue, Process
from datetime import datetime
import os
import traceback

import time

from timeit import default_timer as timer

import running_utils


def read_single_configuration(queue_file, nas_dir, done_filename):
    lines = open(queue_file).readlines()

    if len(lines) == 0:
        raise Exception("Queue is empty!")

    run_set, done_f = running_utils.setup_done_file(nas_dir, done_filename, ["run_type", "training_type", "dataset", "random_seed", "try"])
    done_f.close()

    network = lines.pop(0).rstrip()
    training_type = lines.pop(0).rstrip()
    dataset = lines.pop(0).rstrip()
    random_seed = lines.pop(0).rstrip()
    iTry = lines.pop(0).rstrip()
    source_dir = lines.pop(0).rstrip()
    docker_command = lines.pop(0).rstrip()
    docker_container = lines.pop(0).rstrip()
    docker_image = lines.pop(0).rstrip()
    env = lines.pop(0).rstrip()
    command = lines.pop(0).rstrip()
    eval_command = lines.pop(0).rstrip()
    log_filename = lines.pop(0).rstrip()

    open(queue_file, 'w').writelines(lines)

    runs = []
    main_conf = ('main_run', training_type, dataset, random_seed, int(iTry), network)
    eval_conf = ('eval_run', training_type, dataset, random_seed, int(iTry), network)

    if main_conf not in run_set:
        runs.append((network, training_type, dataset, random_seed, iTry,
                     source_dir, docker_command, docker_container, docker_image, env, 'main_run', command, 'main_' + log_filename))
    if eval_command != '' and eval_conf not in run_set:
        runs.append((network, training_type, dataset, random_seed, iTry,
                     source_dir, docker_command, docker_container, docker_image, env, 'eval_run', eval_command, 'main_' + log_filename))
    return runs


def execute_one_single_gpu_run(global_lockfile, process_lockfile, timestamp, queue_filename, done_filename, working_dir, nas_dir, gpu_index):
    try:
        lock_acquired = False
        while True:
            # get the queue lock
            while not lock_acquired:
                try:
                    os.link(process_lockfile, global_lockfile)
                except:
                    time.sleep(3)
                else:
                    lock_acquired = True

            # get the run
            runs = read_single_configuration(nas_dir + '/' + queue_filename, nas_dir, done_filename)

            # return the lock
            os.unlink(global_lockfile)
            lock_acquired = False

            print("number runs to run: " + str(len(runs)))

            for run in runs:
                (network, training_type, dataset, random_seed, iTry,
                 source_dir, docker_command, docker_container, docker_image, env, command_type, command, log_filename) = run

                container_name = docker_container % gpu_index

                w_dir = working_dir + '/' + network + '_' + training_type + '_' + dataset + '_' + random_seed + '/' + iTry
                docker_command = docker_command + \
                                 ' -v  ' + w_dir + ':/working ' + \
                                 '--name ' + container_name + ' --gpus '"device=%d"'' \
                                 % gpu_index
                com = command

                if command_type == 'main_run':
                    print("create folder, copy content to " + w_dir)
                    #'''
                    subprocess.call('mkdir ' + working_dir + '/' + network + '_' + training_type + '_' + dataset + '_' + random_seed, shell=True)
                    subprocess.call('rm -rf ' + w_dir, shell=True)
                    subprocess.call('cp -r ' + source_dir + '/ ' + w_dir, shell=True)
                    subprocess.call('chmod 777 -R ' + w_dir, shell=True)
                    #'''
                '''
                elif command_type == 'eval_run':
                    print("create folder (eval_run), copy content to " + w_dir)
                    subprocess.call('mkdir -p ' + working_dir + '/' + network + '_' + training_type + '_' + dataset + '_' + random_seed, shell=True)
                    subprocess.call('cp -rT ' + source_dir + '/ ' + w_dir, shell=True)
                    subprocess.call('chmod 777 -R ' + w_dir, shell=True)    
                '''

                # execute the run
                log_filename = timestamp + '_' + command_type + '_' +  log_filename

                if env == '##_NO_CONDA_##':
                    run_command = 'bash /working/1_0_1_execute_single_run_no_conda.sh ' + '"' + com + '" ' + '/working/' + log_filename

                    full_command = docker_command + ' ' + docker_image + ' ' + run_command
                else:

                    run_command = 'bash /working/1_0_execute_single_run.sh ' + env + ' "' + com + '" ' + '/working/' + log_filename

                    full_command = docker_command + ' ' + docker_image + ' ' + 'su -l -c \'' + run_command + '\' userHHH'

                print('RUNNING: ' + full_command)

                begin = timer()
                subprocess.call(full_command, shell=True)
                end = timer()

                subprocess.call('docker container rm ' + container_name, shell=True)

                # get the queue lock
                while not lock_acquired:
                    try:
                        os.link(process_lockfile, global_lockfile)
                    except:
                        time.sleep(3)
                    else:
                        lock_acquired = True

                with open(nas_dir + '/' + done_filename, 'a') as done_f:
                    done_f.write('%s,%s,%s,%s,%s,%s,%.5f\n' % (network, command_type, training_type, dataset, random_seed, iTry, (end - begin)))

                # return the lock
                os.unlink(global_lockfile)
                lock_acquired = False

    except Exception:
        print(traceback.format_exc())
        if lock_acquired:
            os.unlink(global_lockfile)


def main():
    # read the parameter argument parsing
    parser = argparse.ArgumentParser(
        description='Run multiple training run on each of available gpus')
    parser.add_argument('running_dir', help="the path of the running folder")
    parser.add_argument('nas_dir', help="the path of the shared nas folder")
    parser.add_argument('done_filename', help="the done filename")
    parser.add_argument('queue_filename', help="the queue filename")
    parser.add_argument('server', help="the server name")
    args = parser.parse_args()

    # load parameters
    running_dir = args.running_dir
    nas_dir = args.nas_dir
    done_filename = args.done_filename
    queue_filename = args.queue_filename
    server = args.server

    # setup lock
    global_lockfile = nas_dir + '/queue_lock'

    # get timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

    # setup gpu indexes
    if server == 'deepgpu1':
        available_gpus = [0, 1, 2, 3]
    elif server == 'deepgpu2':
        available_gpus = [0, 1, 2, 3]
    elif server == 'deepgpu3':
        available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
        #available_gpus = [0, 2, 4, 6]
    elif server == 'neurips2021':
        available_gpus = [0]
    else:
        raise Exception(server + " is not a supported server")

    # start up processes
    processes = []
    for gpu_index in available_gpus:
        process_lockfile = nas_dir + '/' + server + '.' + str(gpu_index) + '.lock'

        # create the file if necessary (only once, at the beginning of each process)
        with open(process_lockfile, 'w') as f:
            f.write('\n')

        # start each processes
        process = Process(target=execute_one_single_gpu_run,
                          args=(global_lockfile, process_lockfile,
                                timestamp,
                                queue_filename, done_filename,
                                running_dir, nas_dir, gpu_index))
        process.start()
        processes.append(process)

    # wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
