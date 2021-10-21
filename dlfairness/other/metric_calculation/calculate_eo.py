"""
A file to calculate bias metric: Equality of Opportunity (EO)
"""
from utils import *

EO_TP = "equality_of_odds_true_positive"
EO_FP = "equality_of_odds_false_positive"
METRIC = [EO_TP, EO_FP]


def eo(metric):
    """
    A function that iterates all models and calculate given metric
    """
    print(">>>> ", metric)
    # Iterate for each directory
    dir_list = get_dir_list()

    for main_dir in dir_list:
        print(main_dir)
        # Get confusion matrix map to retrieve confusion matrix
        mapping = Mapping(main_dir)
        cm_map = mapping.get_map(post=CM_PATH)
        bias_mapping = BiasMapping(class_index=mapping.class_index, main_path=mapping.path, metric=metric)

        # Calculate metric for each directory
        for path in cm_map:
            bias_list = np.array([])
            idx = 0
            for file in cm_map[path]:
                if "json" in file:
                    with open(path + file) as json_file:
                        data = json.load(json_file)
                    if metric == EO_TP:
                        bias = calculate_tp(data)
                    elif metric == EO_FP:
                        bias = calculate_fp(data)
                    else:
                        print("Wrong metric")
                        return

                    if bias_list.size == 0:
                        bias_list = np.array([bias])
                    else:
                        bias_list = np.append(bias_list, np.array([bias]), axis=0)
                    idx += 1

            bias_mapping.set_df(bias_list, path)

        bias_mapping.save_csv()


def calculate_tp(data):
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
    label_index = 0

    if g > 2:
        print("Need update for multiple protected attributes.")
        return


    # EOTP = |TPR_0 - TPR_1|
    # TPR = TP / P = TP / (TP + FN)
    # Bias_TP = |Pr[Y_hat = 1 | g=0, y=1] - Pr[Y_hat = 1 | g=1, y=1]|
    # group_zero = Pr[Y_hat = 1 | g=0, y=1]
    # group_one = Pr[Y_hat = 1 | g=1, y=1]
    for label in labels:
        group_zero_count = int(df_cm["0"][label]["TP"])
        group_zero_total = int(df_cm["0"][label]["TP"] + df_cm["0"][label]["FN"])
        group_one_count = int(df_cm["1"][label]["TP"])
        group_one_total = int(df_cm["1"][label]["TP"] + df_cm["1"][label]["FN"])

        if group_zero_total != 0:
            group_zero = group_zero_count / group_zero_total
        else:
            group_zero = 0

        if group_one_total != 0:
            group_one = group_one_count / group_one_total
        else:
            group_one = 0

        if group_one != 0:
            res[label_index] = abs(group_zero - group_one)

        label_index += 1

    res[-1] = np.mean(res[:-1])

    return res


def calculate_fp(data):
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
    label_index = 0

    if g > 2:
        print("Need update for multiple protected attributes.")
        return

    # EOFP = |FPR_0 - FPR_1|
    # FPR = FP / N = FP / (FP + TN)
    # Bias_FP = |Pr[Y_hat = 1 | g=0, y=0] - Pr[Y_hat = 1 | g=1, y=0]|
    # group_zero = Pr[Y_hat = 1 | g=0, y=0]
    # group_one = Pr[Y_hat = 1 | g=1, y=0]
    for label in labels:
        group_zero_count = int(df_cm["0"][label]["FP"])
        group_zero_total = int(df_cm["0"][label]["FP"] + df_cm["0"][label]["TN"])
        group_one_count = int(df_cm["1"][label]["FP"])
        group_one_total = int(df_cm["1"][label]["FP"] + df_cm["1"][label]["TN"])

        if group_zero_total != 0:
            group_zero = group_zero_count / group_zero_total
        else:
            group_zero = 0

        if group_one_total != 0:
            group_one = group_one_count / group_one_total
        else:
            group_one = 0

        if group_one != 0:
            res[label_index] = abs(group_zero - group_one)

        label_index += 1

    res[-1] = np.mean(res[:-1])

    return res


if __name__ == '__main__':
    for m in METRIC:
        eo(m)