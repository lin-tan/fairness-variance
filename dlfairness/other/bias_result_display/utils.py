import yaml
import scipy.stats
import statistics
import math

from pathlib import Path
from functools import partial

from yaml import loader

import decimal_utils
from abbrs import prefix_abbr, setting_abbr, metric_abbr

def write_tex_table_line(f, line):
    s = " & ".join(line)
    f.write(s)
    f.write('  \\\\\n')


def stat_significance(val_1, val_2, p_value):
    if p_value > 0.05:
        return 'NS'
    elif val_1 < val_2:
        return '+'
    else:
        return '-'

def stat_threshold_formatter(s, thres=0.01, decimal_points=2):
    if isinstance(s, str):
        s_n = float(s)
    else:
        s_n = s
    
    if math.isclose(s_n, 0):
        f_str = '{:.' + str(decimal_points) + 'f}'
        return f_str.format(s_n)

    if s_n < thres:
        return "<0.01"
    else:
        f_str = '{:.' + str(decimal_points) + 'f}'
        return f_str.format(s_n)

def stat_threshold_formatter_percentage(s, thres=0.1, decimal_points=1):
    if isinstance(s, str):
        s_n = float(s) * 100
    else:
        s_n = s * 100
    
    if math.isclose(s_n, 0):
        f_str = '{:.' + str(decimal_points) + 'f}'
        return f_str.format(s_n)

    if s_n < thres:
        return "<0.1"
    else:
        f_str = '{:.' + str(decimal_points) + 'f}'
        return f_str.format(s_n)


def stat_formatter(s, decimal_points=2):
    if isinstance(s, str):
        s_n = float(s)
    else:
        s_n = s
    f_str = '{:.' + str(decimal_points) + 'f}'
    return f_str.format(s_n)

def stat_formatter_percentage(s, decimal_points=1):
    if isinstance(s, str):
        s_n = float(s) * 100
    else:
        s_n = s * 100
    f_str = '{:.' + str(decimal_points) + 'f}'
    return f_str.format(s_n)

def cohen_d(c0, c1):
    cohens_d = (statistics.mean(c0) - statistics.mean(c1)) / (math.sqrt((statistics.stdev(c0) ** 2 + statistics.stdev(c1) ** 2) / 2))
    return cohens_d

def ndi_mapper(metric_key):
    if metric_key == 'DI':
        return '\\ndi'
    else:
        return metric_key

def ndi_mapper_csv(metric_key):
    if metric_key == 'DI':
        return 'DI bar'
    else:
        return metric_key
