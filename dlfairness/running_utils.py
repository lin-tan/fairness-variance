from pathlib import Path
import pandas as pd
import ntpath
import os


def create_directory(filepath):
    dir = ntpath.dirname(filepath)
    os.makedirs(dir, exist_ok=True)

def chmod_directory(dirpath):
    os.os.chmod(dirpath, 0o777)


def get_done_set(result_dir, filename, special_cols):
    done_path = result_dir + '/' + filename
    done_data = pd.read_csv(done_path, skipinitialspace=True)

    # Run list
    networks = done_data['network'].values
    special_cols_data = [done_data[col].values for col in special_cols]
    special_cols_data.append(networks)

    run_set = [tuple([col_data[r] for col_data in special_cols_data]) for r in range(len(networks))]

    print('Run ' + str(len(networks)) + ' projects!')

    return run_set

def setup_done_file(result_dir, filename, special_cols):
    done_path = result_dir + '/' + filename
    done_file = Path(done_path)
    if not done_file.is_file():
        run_set = []

        done_out_f = open(done_path, "w")
        done_out_f.write('network')
        for col in special_cols:
            done_out_f.write(',%s' % col)
        done_out_f.write(',time\n')
    else:
        run_set = get_done_set(result_dir, filename, special_cols)

        done_out_f = open(done_path, "a")

    return set(run_set), done_out_f
