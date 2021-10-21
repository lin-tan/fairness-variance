"""
A file to calculate bias metric: Bias Amplification (BA)
"""
from utils import *

METRIC = "bias_amplification"


def ba():
    """
    A function that iterates all models and calculate given metric
    """
    # Iterate for each directory
    dir_list = get_dir_list()

    for main_dir in dir_list:
        print(main_dir)
        # Get confusion matrix map to retrieve confusion matrix
        mapping = Mapping(main_dir)
        cm_map = mapping.get_map(post=CM_PATH)
        bias_mapping = BiasMapping(class_index=mapping.class_index, main_path=mapping.path, metric=METRIC)

        # Calculate bias amplification for each directory
        for path in cm_map:
            bias_list = np.array([])
            idx = 0
            for file in cm_map[path]:
                if "json" in file:
                    with open(path + file) as json_file:
                        data = json.load(json_file)
                    bias = calculate(data)
                    if bias_list.size == 0:
                        bias_list = np.array([bias])
                    else:
                        bias_list = np.append(bias_list, np.array([bias]), axis=0)
                    idx += 1

            bias_mapping.set_df(bias_list, path)

        bias_mapping.save_csv()


def calculate(data):
    protected_groups = list(data.keys())
    labels = data[protected_groups[0]].keys()

    df_cm = {}
    for group in protected_groups:
        df_cm[group] = pd.DataFrame.from_dict(data[group])

    c = len(labels)
    G = len(protected_groups)
    res = np.zeros(c + 1)
    label_index = 0

    # BA = |(TP_g'+FP_g') / (TP+FP) - P_g'/P|, g' = argmax_g P_g/P
    for label in labels:
        group_count = pd.Series(index=protected_groups)
        group_positive_count = pd.Series(index=protected_groups)
        for group in protected_groups:
            group_count[group] = int(df_cm[group][label]["TP"] + df_cm[group][label]["FP"])
            group_positive_count[group] = int(df_cm[group][label]["TP"] + df_cm[group][label]["FN"])

        if group_count.sum() != 0 and group_positive_count.sum() != 0:
            b_o_g = np.divide(group_count, group_count.sum())
            p_g_over_p = np.divide(group_positive_count, group_positive_count.sum())
        else:
            res[label_index] = 0
            continue

        for g_idx in range(G):
            if p_g_over_p[g_idx] == 1/G:
                res[label_index] = np.max(b_o_g) - 0.5
                break
            elif group_positive_count[g_idx] > 1/G:
                res[label_index] = abs(b_o_g[g_idx] - p_g_over_p[g_idx])

        label_index += 1

    res[-1] = np.mean(res[:-1])

    return res


if __name__ == '__main__':
    ba()