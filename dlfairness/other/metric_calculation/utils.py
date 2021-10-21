"""
A file that contains utility functions used in metric calculation
"""
import os
import sys
import ast
import json
import copy
import pandas as pd
import numpy as np
import decimal
from decimal import Decimal, localcontext
from builtins import any as b_any

PROTECTED_GROUP = "protected_label"
GROUND_TRUTH = "ground_truth"
PREDICTED_LABEL = "prediction_result"
IDX = "idx"
TRUE = 1
FALSE = 0
GLOBAL_ROUND_DIGIT = 4
RUN_MAX = 16
NOTATION = '{0:.3e}'

OUTPUT_PATH = "_output"
CM_PATH = OUTPUT_PATH + "/cm"
RESULT_PATH = OUTPUT_PATH + "/result"
DATA_PATH = "data/"
COUNT_PATH = OUTPUT_PATH + "/count"
DATA_OUTPUT_PATH = OUTPUT_PATH + "/data"
COUNT = "count_test"
NOT_INCLUDED = ["DS_STORE", "pycache", OUTPUT_PATH, DATA_PATH[:-1], "unit_test"]

MAX = "max"
MIN = "min"
MAX_DIFF = "max_diff"
MEAN = "mean"
STD_DEV = "std_dev"
REL_MAX_DIFF = "rel_maxdiff(%)"
CATEGORY = [MAX_DIFF, MAX, MIN, STD_DEV, MEAN, REL_MAX_DIFF]


def get_file_map(dir_path):
    """
    A function to create a map of entire directory structures for each model

    :param dir_path: A path of directory that contains prediction result
    :return: A map of directory structures
             - key: each model's main directory path
             - value: path of each CSV file
    """
    file_map = {}
    dir_path = "./prediction/" + dir_path[2:]

    for dir_name, subdir_list, file_list in os.walk(dir_path):
        dir_name += '/'
        for fileName in file_list:
            if b_any("csv" in x for x in file_list) and dir_name != dir_path and "output" not in dir_name:
                dir_name = dir_name.replace("/prediction/", "/")
                if dir_name not in file_map:
                    file_map.update({dir_name: []})
                if "csv" in fileName:
                    file_map[dir_name].append(fileName)

    return file_map


class Mapping:
    """
    A class to manage file mapping for each model output
    """
    def __init__(self, path="."):
        self.path = path
        self.post_map = {}
        self.class_number = 0
        self.class_index = []

    def get_map(self, post):
        """
        A function to create a file map of entire directory structures for each model output

        :param post: A postfix to add to main path
        :return: A map of directory structures
                 - key: each model's output path
                 - value: path of each JSON file
        """
        dir_path = self.path[:-1] + post
        for dir_name, subdir_list, file_list in os.walk(dir_path):
            dir_name += '/'
            if dir_name != dir_path:
                run_idx = 0
                file_list.sort()
                for fileName in file_list:
                    if b_any("json" in x for x in file_list):
                        if dir_name not in self.post_map:
                            self.post_map.update({dir_name: []})
                        if ".json" in fileName:
                            self.post_map[dir_name].append(fileName)
                            run_idx += 1
                    if run_idx >= RUN_MAX:
                        break

        self.get_class_number()
        self.get_class_index()

        return self.post_map

    def get_class_index(self):
        """
        A function to set index for each class into a list, and add overall as a last element
        """
        if self.class_number == 0:
            self.get_class_number()
        digit = len(str(self.class_number))
        for i in range(self.class_number):
            self.class_index.append(f"class{i:{digit}.0f}")
        self.class_index.append("overall")

    def get_class_number(self):
        """
        A function to get the number of classes
        """
        (k, v) = list(self.post_map.items())[0]
        with open(str(k) + str(v[0])) as json_file:
            data = json.load(json_file)
        group, group_ex = data.popitem()
        self.class_number = len(group_ex)


def copy_directory(file_map, post):
    """
    A function to copy directory structure and add postfix to root path

    :param file_map: A map of file structure for each model prediction result
    :param post: A postfix to add to the root path
    """
    path_list = list(file_map.keys())
    for path in path_list:
        main_dir = path.split("/")[1]
        new_path = path.replace(main_dir, main_dir + post)

        if not os.path.exists(new_path):
            os.makedirs(new_path)


def get_new_path(path, post):
    """
    A function to get a new path with given postfix

    :param path: A directory path
    :param post: A postfix
    :return: A new path with postfix added
    """
    main_dir = path.split("/")[1]
    new_path = path.replace(main_dir, main_dir + post)

    return new_path


def print_map(data):
    """
    A function to print dictionary value with indentation

    :param data: A dictionary to be printed
    """
    print(json.dumps(data, indent=2))


def dir_check(dir_name):
    """
    A function to check whether a given directory path exist
    If not, create a directory.

    :param dir_name: A directory path
    """
    if dir_name[-1] != "/":
        dir_name += "/"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def get_dir_list(dir_path="."):
    """
    A function to get a list of directories in a given path
    It only counts a directory that contains prediction result of models

    :param dir_path: A path which the function is trying to get directory list
    :return: A list of directories that contain prediction result of models
    """
    dir_path = dir_path + "/prediction/"
    name_list = os.listdir(dir_path)
    dir_list = []

    for name in name_list:
        name_path = "./prediction/" + name
        if os.path.isdir(name_path) and not any(x in name for x in NOT_INCLUDED):
            dir_list.append("./" + name + "/")

    return dir_list


def round_significant_digit(num, digit=GLOBAL_ROUND_DIGIT):
    """
    A function to round a float value into given significant digit

    :param num: A float value
    :param digit: A significant digit value
    :return: A rounded float value
    """
    with localcontext() as ctx:
        ctx.prec = digit
        return ctx.create_decimal(num)


def round_significant_format(num, digit=GLOBAL_ROUND_DIGIT):
    """
    A function to round and format a float value into given significant digit

    :param num: A float value
    :param digit: A significant digit value
    :return: A rounded float value
    """
    with localcontext() as ctx:
        ctx.prec = digit
        t = ctx.create_decimal(num)
        format_str = '#.' + str(digit) + 'g'
        return format(float(t), format_str)


def round_list(tlist, digit=GLOBAL_ROUND_DIGIT):
    """
    A function to round a list of float values into given significant digit

    :param tlist: A list of float values
    :param digit: A significant digit value
    :return: A list of rounded float values
    """
    if not isinstance(tlist, list) and not isinstance(tlist, np.ndarray):
        return tlist
    return [round_significant_digit(e, digit=digit) for e in tlist]


class BiasMapping:
    """
    A class to manage bias metric values mapping for each model output
    """
    def __init__(self, class_index, main_path, metric):
        self.bias_map_round = {}
        self.bias_map_notation = {}
        self.raw = {}
        self.class_index = class_index
        self.main_path = main_path
        self.metric = metric

    def set_df(self, bias_list, path):
        """
        A function to set a dataframe that contains aggregated bias metric value

        :param bias_list: A dataframe of raw bias metric values
        :param path: A path including model name
        """
        self.raw[path.split(CM_PATH)[-1][1:]] = bias_list.tolist()

        bias_max = np.amax(bias_list, axis=0)
        bias_min = np.amin(bias_list, axis=0)
        bias_max_diff = np.subtract(bias_max, bias_min)
        bias_mean = np.mean(bias_list, axis=0)
        bias_std_dev = np.std(bias_list, axis=0)
        bias_rel_maxdiff = np.divide(bias_max_diff, bias_mean, out=np.zeros_like(bias_max_diff), where=bias_mean != 0) * 100

        df_bias_part = pd.DataFrame(index=self.class_index, columns=CATEGORY)
        df_bias_part[MAX] = round_list(bias_max)
        df_bias_part[MIN] = round_list(bias_min)
        df_bias_part[MAX_DIFF] = round_list(bias_max_diff)
        df_bias_part[MEAN] = round_list(bias_mean)
        df_bias_part[STD_DEV] = round_list(bias_std_dev)
        df_bias_part[REL_MAX_DIFF] = round_list(bias_rel_maxdiff)

        self.bias_map_round[path.replace(CM_PATH, "")] = df_bias_part

    def save_csv(self):
        """
        A function to save bias mapping into a CSV file
        """
        df_bias_round = pd.concat(self.bias_map_round).sort_index(axis=0)

        output_path = get_new_path(self.main_path, RESULT_PATH)
        dir_check(output_path)
        df_bias_round.to_csv(output_path + self.metric + ".csv")

        with open(output_path + self.metric + "_raw.json", "w") as fp:
            json.dump(self.raw, fp, indent=2)