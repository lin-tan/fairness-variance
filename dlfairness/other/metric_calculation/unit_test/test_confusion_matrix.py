import unittest
from confusion_matrix import *

CM_FILENAME = "sample_output/cm/work1/try00.json"
CM2_FILENAME = "sample2_output/cm/work1/try00.json"


class TestConfusionMatrix(unittest.TestCase):
    @classmethod
    # Set current path to working directory
    def setUpClass(cls):
        os.chdir("sample_for_test/")

    # Test `cm()` from confusion_matrix.py
    #   `cm()` calculates confusion matrix from given data
    def test_cm(self):
        print(">>>> Test confusion_matrix.py with sample")

        # Delete previous confusion matrix
        if os.path.exists(CM_FILENAME):
            os.remove(CM_FILENAME)
            print("Previous confusion matrix deleted")

        # Run `cm()` and load confusion matrix as dictionary
        cm()
        with open(CM_FILENAME, 'r') as outfile:
            actual_cm = json.load(outfile)

        # Set expected confusion matrix according to data
        expected_cm = {
            # For each protected group
            "0": {
                # For each ground truth, set confusion matrix
                "0": {"TP": 3, "FP": 2, "TN": 5, "FN": 1},
                "1": {"TP": 5, "FP": 1, "TN": 3, "FN": 2}
            },
            "1": {
                "0": {"TP": 5, "FP": 1, "TN": 2, "FN": 1},
                "1": {"TP": 2, "FP": 1, "TN": 5, "FN": 1}
            }
        }

        # Compare expected and actual dictionary
        print("-- Expected: ")
        print(expected_cm)
        print("-- Actual")
        print(json.dumps(actual_cm))
        print("\n\n")

        self.assertDictEqual(expected_cm, actual_cm)

    # Test `cm()` from confusion_matrix.py
    #   `cm()` calculates confusion matrix from given data
    def test_cm2(self):
        print(">>>> Test confusion_matrix.py with sample2")

        # Delete previous confusion matrix
        if os.path.exists(CM2_FILENAME):
            os.remove(CM2_FILENAME)
            print("Previous confusion matrix deleted")

        # Run `cm()` and load confusion matrix as dictionary
        cm()
        with open(CM2_FILENAME, 'r') as outfile:
            actual_cm = json.load(outfile)

        # Set expected confusion matrix according to data
        expected_cm = {
            # For each protected group
            "0": {
                # For each ground truth, set confusion matrix
                "0": {"TP": 3, "FP": 1, "TN": 4, "FN": 3},
                "1": {"TP": 3, "FP": 3, "TN": 1, "FN": 4}
            },
            "1": {
                "0": {"TP": 2, "FP": 1, "TN": 4, "FN": 2},
                "1": {"TP": 3, "FP": 2, "TN": 1, "FN": 3}
            }
        }

        # Compare expected and actual dictionary
        print("-- Expected: ")
        print(expected_cm)
        print("-- Actual")
        print(json.dumps(actual_cm))

        self.assertDictEqual(expected_cm, actual_cm)


if __name__ == '__main__':
    unittest.main()
