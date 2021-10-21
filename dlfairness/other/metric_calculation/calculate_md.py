"""
A file to calculate bias metric: Demographic Parity (DP)
"""
from utils import *

METRIC = "mean_difference_score"


def md():
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
    for group in protected_groups:
        df_cm[group] = pd.DataFrame.from_dict(data[group])

    c = len(labels)
    res = np.zeros(c + 1)
    g = len(protected_groups)
    label_index = 0

    if g > 2:
        print("Need update for multiple protected attribute.")
        return

    # DP = |PPR_0 - PPR_1|
    # PPR = (TP + FP) / (P+N)
    for label in labels:
        group_zero_count = int(df_cm["0"][label]["TP"] + df_cm["0"][label]["FP"])
        group_zero_total = df_cm["0"].to_numpy().sum() / c
        group_one_count = int(df_cm["1"][label]["TP"] + df_cm["1"][label]["FP"])
        group_one_total = df_cm["1"].to_numpy().sum() / c

        if group_zero_total != 0:
            group_zero = group_zero_count / group_zero_total
        else:
            group_zero = 0

        if group_one_total != 0:
            group_one = group_one_count / group_one_total
        else:
            group_one = 0

        res[label_index] = abs(group_zero - group_one)

        label_index += 1

    res[-1] = np.mean(res[:-1])

    return res


if __name__ == '__main__':
    md()