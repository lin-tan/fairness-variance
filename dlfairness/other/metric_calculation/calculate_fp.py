"""
A file to calculate bias metric: False Positive Subgroup Fairness (FPSF)
"""
from utils import *

METRIC = "false_positive_subgroup_fairness"


def fp():
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

        # Calculate metric for each directory
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
    total = 0
    for group in protected_groups:
        df_cm[group] = pd.DataFrame.from_dict(data[group])
        total += int(df_cm[group].values.sum())

    c = len(labels)
    res = np.zeros(c + 1)
    g = len(protected_groups)
    fp_dg = np.zeros(g)
    group_false = np.zeros(g)
    total = total / c
    label_index = 0

    # FPSF = (sum of Pr_g) * |FPR - FPR_g|
    # FPR = FP / N
    # alpha: list of Pr_g
    # fp_d: FPR
    # fp_dg: FPR for each g(protected group)
    for label in labels:
        fp_d = 0
        group_idx = 0
        for group in protected_groups:
            group_false[group_idx] = int(df_cm[group][label]["TN"] + df_cm[group][label]["FP"])
            group_fp = int(df_cm[group][label]["FP"])
            fp_d += group_fp
            fp_dg[group_idx] = group_fp / group_false[group_idx]
            group_idx += 1
        alpha = np.divide(group_false, total)
        fp_d = fp_d / group_false.sum()

        res[label_index] = np.multiply(alpha, abs(fp_d - fp_dg)).sum()
        label_index += 1

    res[-1] = np.mean(res[:-1])

    return res


if __name__ == '__main__':
    fp()