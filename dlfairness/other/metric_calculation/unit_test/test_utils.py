import unittest
from utils import *
from confusion_matrix import *
import decimal
import pandas.testing as pd_testing

MAIN_DIR_PATH = "./sample/"
SAMPLE_OUTPUT_PATH = "./sample_output/cm/"


class TestUtils(unittest.TestCase):
    @classmethod
    # Set current path to working directory
    def setUpClass(cls):
        os.chdir("sample_for_test/")

    # Test `set_df(self, bias_list, path)`
    #   Calculates max, min, max_diff, mean, std_dev, rel_max_diff
    #   among entire trials of each class/overall
    def test_set_df(self):
        # Get result from the method
        mapping = Mapping(MAIN_DIR_PATH)
        bias_mapping = BiasMapping(class_index=mapping.class_index, main_path=mapping.path, metric="metric")
        bias_list = np.array([[0.1234, 0.2345, 0.3456],
                              [0.4567, 0.5678, 0.6789]])
        bias_mapping.set_df(bias_list, SAMPLE_OUTPUT_PATH)
        actual_df = bias_mapping.bias_map_round[MAIN_DIR_PATH]

        # Set expected values
        trial = {MAX_DIFF: [0.3333, 0.3333, 0.3333],
                 MAX: [0.4567, 0.5678, 0.6789],
                 MIN: [0.1234, 0.2345, 0.3456],
                 STD_DEV: [0.16665, 0.16665, 0.16665],
                 MEAN: [0.29005, 0.40115, 0.51225],
                 REL_MAX_DIFF: [114.9, 83.09, 65.07]}

        # Convert each value to dtype(decimal) to make sure it has the same dtype
        # (set_df uses round_list, which converts all values to decimal dtype)
        # Round each value to 4 digits (except for relative max difference)
        DIGIT = decimal.Decimal("0.0001")
        decimal_trial = {}
        for category in trial:
            decimal_trial[category] = [decimal.Decimal(x).quantize(DIGIT) for x in trial[category]]

        # Compare expected and actual dataframe
        print("-- Expected: ")
        print(pd.DataFrame(decimal_trial))
        print("\n-- Actual")
        print(actual_df)
        pd_testing.assert_frame_equal(pd.DataFrame(decimal_trial), actual_df)

    def test_set_df_2(self):
        mapping = Mapping(MAIN_DIR_PATH)
        bias_mapping = BiasMapping(class_index=mapping.class_index, main_path=mapping.path, metric="metric")
        bias_list = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        bias_mapping.set_df(bias_list, SAMPLE_OUTPUT_PATH)

        trial_max = np.amax(bias_list, axis=0)
        trial_min = np.amin(bias_list, axis=0)
        trial_max_diff = np.subtract(trial_max, trial_min)
        trial_mean = np.mean(bias_list, axis=0)
        trial_std_dev = np.std(bias_list, axis=0)
        trial_rel_maxdiff = np.divide(trial_max_diff, trial_mean, out=np.zeros_like(trial_max_diff),
                                      where=trial_mean != 0) * 100

        round_max = round_list(trial_max)
        round_min = round_list(trial_min)
        round_max_diff = round_list(trial_max_diff)
        round_mean = round_list(trial_mean)
        round_std_dev = round_list(trial_std_dev)
        round_rel_max_diff = round_list(trial_rel_maxdiff)

        for index, row in bias_mapping.bias_map_round[MAIN_DIR_PATH].iterrows():
            self.assertEqual(round_max[index], row[MAX])
            self.assertEqual(round_min[index], row[MIN])
            self.assertEqual(round_max_diff[index], row[MAX_DIFF])
            self.assertEqual(round_mean[index], row[MEAN])
            self.assertEqual(round_std_dev[index], row[STD_DEV])
            self.assertEqual(round_rel_max_diff[index], row[REL_MAX_DIFF])

    def test_map(self):
        print(get_dir_list())
        print(get_file_map(MAIN_DIR_PATH))
        cm()
        map = Mapping(MAIN_DIR_PATH)
        print(map.get_map(CM_PATH))
        print("Class Index: " + str(map.class_index))
        print("Number of classes: " + str(map.class_number))



if __name__ == '__main__':
    unittest.main()
