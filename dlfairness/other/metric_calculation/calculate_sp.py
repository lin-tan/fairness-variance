"""
A file to calculate bias metric: Bias Amplification (BA)
"""
from utils import *

METRIC = "statistical_parity"


def sp():
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
    alpha = np.zeros(g)
    sp_dg = np.zeros(g)
    total = total / c
    label_index = 0

    # SPSF = (sum of Pr_g) * |PPR - PPR_g|
    # alpha: list of Pr_g
    # sp_d: PPR
    # sp_dg: PPR for each g(protected group)
    # PPR = (TP + FP) / (P+N)
    for label in labels:
        sp_d = 0
        group_idx = 0
        for group in protected_groups:
            group_total = df_cm[group].values.sum() / c
            alpha[group_idx] = group_total / total
            group_acceptance = int(df_cm[group][label]["TP"] + df_cm[group][label]["FP"])
            sp_d += group_acceptance
            sp_dg[group_idx] = group_acceptance / group_total
            group_idx += 1
        sp_d = sp_d / total

        res[label_index] = np.multiply(alpha, abs(sp_d - sp_dg)).sum()
        label_index += 1

    res[-1] = np.mean(res[:-1])

    return res


if __name__ == '__main__':
    sp()