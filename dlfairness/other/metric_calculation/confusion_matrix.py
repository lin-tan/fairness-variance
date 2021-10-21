"""
A file to create confusion matrix for each classification result
Example : {
    # For each protected group
    "0": {
        # For each ground truth, create confusion matrix
        "0": {"TP": 3, "FP": 2, "TN": 5, "FN": 1},
        "1": {"TP": 5, "FP": 1, "TN": 3, "FN": 2}
    },
    "1": {
        "0": {"TP": 5, "FP": 1, "TN": 2, "FN": 1},
        "1": {"TP": 2, "FP": 1, "TN": 5, "FN": 1}
    }
}
"""
from utils import *


def cm(path="."):
    """
    A main function to create confusion matrix

    :param path: A path of root directory that contains prediction result of all models
    """
    # Iterate for each directory
    dir_list = get_dir_list(path)


    for main_dir in dir_list:
        print(main_dir)
        # Create directory to store confusion matrices
        file_map = get_file_map(main_dir)

        if file_map == {}:
            print(">>> No CSV directories to map.")
            return
        copy_directory(file_map, CM_PATH)
        copy_directory(file_map, COUNT_PATH)

        total_dir = len(file_map)
        dir_idx = 1

        # Traverse each file to create confusion matrix JSON
        for path in file_map:
            print("---- %d/%d: %s" % (dir_idx, total_dir, path))
            dir_idx += 1
            cm_path = get_new_path(path, CM_PATH)
            count_path = get_new_path(path, COUNT_PATH)
            for f in file_map[path]:
                filename = "./prediction" + path[1:] + f

                df = pd.read_csv(filename)
                cm_filename = (cm_path + f).split(".csv")[0]
                count_filename = (count_path + f).split(".csv")[0]

                # Get labels of each column
                ground_truth = pd.get_dummies(df[GROUND_TRUTH]).columns.tolist()
                predicted_label = pd.get_dummies(df[PREDICTED_LABEL]).columns.tolist()
                protected_groups = pd.get_dummies(df[PROTECTED_GROUP]).columns.tolist()

                # Check label property
                sample = df[GROUND_TRUTH][0]
                sample_type = type(sample)

                if sample_type == str:
                    df_count, count_res = count_multi(df)
                    cm_res = calculate_multi(ast.literal_eval(sample), protected_groups, count_res)
                elif sample.dtype.kind == "i":
                    df_count, count_res = count(df)
                    cm_res = calculate(ground_truth, protected_groups, df_count)
                elif sample.dtype == np.floating:   # Added to handle CSV with np float
                    df_changed = df[df.columns].fillna(0.0).astype(int)
                    ground_truth = pd.get_dummies(df_changed[GROUND_TRUTH]).columns.tolist()
                    protected_groups = pd.get_dummies(df_changed[PROTECTED_GROUP]).columns.tolist()
                    df_count, count_res = count(df_changed)
                    cm_res = calculate(ground_truth, protected_groups, df_count)
                else:
                    print("Wrong format")
                    return

                with open(count_filename + '.json', 'w') as outfile:
                    json.dump(count_res, outfile, indent=2)

                with open(cm_filename + '.json', 'w') as outfile:
                    json.dump(cm_res, outfile, indent=2)


def count(df):
    """
    A function to count numbers for each prediction cases

    :param df: A dataframe of raw prediction result
    :return:
            - df_count
                Column: Ground Truth
                Row: Predicted label
                TP: df_count[ground truth label, ground truth label]
                FP: entire row except df_count[ground truth label, ground truth label]
                FN: entire column except df_count[ground truth label, ground truth label]
                TN: remaining
            - count_res
                A dictionary format of df_count
    """
    # Get labels of each column
    ground_truth = pd.get_dummies(df[GROUND_TRUTH]).columns.tolist()
    protected_groups = pd.get_dummies(df[PROTECTED_GROUP]).columns.tolist()

    # Set dictionary for each protected group
    group_inner_dic = {i: 0 for i in ground_truth}
    group_dic = {j: copy.deepcopy(group_inner_dic) for j in ground_truth}

    df_count = {}
    count_res = {}

    # Count cases for each protected group label
    for group in protected_groups:
        matrix = copy.deepcopy(group_dic)
        group_df = df[df[PROTECTED_GROUP] == group]
        for truth in ground_truth:
            ground_df = group_df[group_df[GROUND_TRUTH] == truth]
            count = ground_df[PREDICTED_LABEL].value_counts()
            for predicted in ground_truth:
                if predicted in count:
                    matrix[truth][predicted] = int(count[predicted])
        df_count[group] = pd.DataFrame.from_dict(matrix)
        count_res[group] = matrix

        df.drop(group_df.index, inplace=True)

    return df_count, count_res


def count_multi(df):
    """
    A function to count numbers for each prediction cases
    This is a case when a model is for multi-label classification.

    :param df: A dataframe of raw prediction result
    :return:
            - df_count
                Column: Ground Truth
                Row: Predicted label
                TP: df_count[ground truth label, ground truth label]
                FP: entire row except df_count[ground truth label, ground truth label]
                FN: entire column except df_count[ground truth label, ground truth label]
                TN: remaining
            - count_res
                A dictionary format of df_count
    """
    # Get labels of each column
    ground_truth_len = len(ast.literal_eval(df[GROUND_TRUTH][0]))
    protected_groups = pd.get_dummies(df[PROTECTED_GROUP]).columns.tolist()
    ground_truth = {}
    predicted_label = {}

    for idx, row in df.iterrows():
        ground_truth[(row[IDX])] = ast.literal_eval(row[GROUND_TRUTH])
        predicted_label[(row[IDX])] = ast.literal_eval(row[PREDICTED_LABEL])

    ground_truth_df = pd.DataFrame.from_dict(ground_truth, orient="index")
    predicted_label_df = pd.DataFrame.from_dict(predicted_label, orient="index")

    # Set dictionary for each protected group
    group_innermost_dic = {i: 0 for i in range(2)}
    group_inner_dic = {i: copy.deepcopy(group_innermost_dic) for i in range(2)}
    group_dic = {j: copy.deepcopy(group_inner_dic) for j in range(ground_truth_len)}

    df_count = {}
    count_res = {}

    # Count cases for each protected group label
    for group in protected_groups:
        matrix = copy.deepcopy(group_dic)
        group_df_idx = list(df[df[PROTECTED_GROUP] == group][IDX])
        for class_idx in range(ground_truth_len):
            for idx in group_df_idx:
                ground = ground_truth_df.at[idx, class_idx]
                predicted = predicted_label_df.at[idx, class_idx]
                matrix[class_idx][ground][predicted] += 1

        df_count[group] = pd.DataFrame.from_dict(matrix)
        count_res[group] = matrix

    return df_count, count_res


def calculate(ground_truth, protected_groups, df_count):
    """
    A function to calculate a confusion matrix

    :param ground_truth: A list of ground truth value
    :param protected_groups: A list of protected group value
    :param df_count: A dataframe that contains prediction summary
    :return: confusion matrix based on df_count
    """
    # Calculate confusion matrix for each protected group label
    cm_res = {}
    cm_inner_dic = {
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0
    }
    cm_dic = {k: copy.deepcopy(cm_inner_dic) for k in ground_truth} ## Added to handle numpy datatype

    for group in protected_groups:
        cm = copy.deepcopy(cm_dic)
        group_count = df_count[group]
        total = int(group_count.values.sum())
        for truth in ground_truth:
            cm[truth]["TP"] = int(group_count.iloc[truth][truth])
            cm[truth]["FP"] = int(group_count.iloc[truth].sum()) - cm[truth]["TP"]
            cm[truth]["FN"] = int(group_count[truth].sum()) - cm[truth]["TP"]
            cm[truth]["TN"] = total - cm[truth]["TP"] - cm[truth]["FP"] - cm[truth]["FN"]
        cm_res[group] = cm

    return cm_res


def calculate_multi(ground_truth, protected_groups, count_res):
    """
    A function to calculate a confusion matrix
    This is a case when a model is for multi-label classification.

    :param ground_truth: A list of ground truth value
    :param protected_groups: A list of protected group value
    :param df_count: A dataframe that contains prediction summary
    :return: confusion matrix based on df_count
    """
    ground_truth_len = len(ground_truth)
    # Calculate confusion matrix for each protected group label
    cm_res = {}
    cm_inner_dic = {
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0
    }
    cm_dic = {k: copy.deepcopy(cm_inner_dic) for k in range(ground_truth_len)}

    for group in protected_groups:
        cm = copy.deepcopy(cm_dic)
        group_count = count_res[group]
        for label in range(ground_truth_len):
            cm[label]["TP"] = int(group_count[label][TRUE][TRUE])
            cm[label]["FP"] = int(group_count[label][FALSE][TRUE])
            cm[label]["FN"] = int(group_count[label][TRUE][FALSE])
            cm[label]["TN"] = int(group_count[label][FALSE][FALSE])
        cm_res[group] = cm

    return cm_res


if __name__ == '__main__':
    cm()