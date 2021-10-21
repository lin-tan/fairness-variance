import os
import unittest
import numpy as np
import numpy.testing as np_testing
from calculate_md import calculate as calculate_md
from calculate_di import calculate as calculate_di
from calculate_sp import calculate as calculate_sp
from calculate_fp import calculate as calculate_fp
from calculate_eo import calculate_fp as calculate_eofp
from calculate_eo import calculate_tp as calculate_eotp
from calculate_ba import calculate as calculate_ba

MAIN_DIR_PATH = "/sample2/"

test_cm = {
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


class TestCalculateMetric2(unittest.TestCase):
    @classmethod
    # Set current path to working directory
    def setUpClass(cls):
        os.chdir("sample_for_test/")

    def test_calculate_md(self):
        print(">>>> Test calculate_md(dp)")

        expected_value = np.array([0.03030303, 0.01010101, 0.02020202])
        actual_value = calculate_md(test_cm)

        print("-- Expected: ")
        print(expected_value)
        print("-- Actual")
        print(actual_value)
        print()

        # Compare each value up to 9th precision
        np_testing.assert_array_almost_equal(expected_value, actual_value, decimal=9)

    def test_calculate_di(self):
        print(">>>> Test calculate_di")

        expected_value = np.array([0.083333333, 0.018181818, 0.050757576])
        actual_value = calculate_di(test_cm)

        print("-- Expected: ")
        print(expected_value)
        print("-- Actual")
        print(actual_value)
        print()

        # Compare each value up to 9th precision
        np_testing.assert_array_almost_equal(expected_value, actual_value, decimal=9)

    def test_calculate_sp(self):
        print(">>>> Test calculate_sp")

        expected_value = np.array([0.015, 0.005, 0.01])
        actual_value = calculate_sp(test_cm)

        print("-- Expected: ")
        print(expected_value)
        print("-- Actual")
        print(actual_value)
        print()

        # Compare each value up to 9th precision
        np_testing.assert_array_almost_equal(expected_value, actual_value, decimal=9)

    def test_calculate_fp(self):
        print(">>>> Test calculate_fp")

        expected_value = np.array([0, 0.014285714, 0.007142857])
        actual_value = calculate_fp(test_cm)

        print("-- Expected: ")
        print(expected_value)
        print("-- Actual")
        print(actual_value)
        print()

        # Compare each value up to 9th precision
        np_testing.assert_array_almost_equal(expected_value, actual_value, decimal=9)

    def test_calculate_eofp(self):
        print(">>>> Test calculate_eofp")

        expected_value = np.array([0, 0.083333333, 0.041666667])
        actual_value = calculate_eofp(test_cm)

        print("-- Expected: ")
        print(expected_value)
        print("-- Actual")
        print(actual_value)
        print()

        # Compare each value up to 9th precision
        np_testing.assert_array_almost_equal(expected_value, actual_value, decimal=9)

    def test_calculate_eotp(self):
        print(">>>> Test calculate_eotp")

        expected_value = np.array([0, 0.071428571, 0.035714286])
        actual_value = calculate_eotp(test_cm)

        print("-- Expected: ")
        print(expected_value)
        print("-- Actual")
        print(actual_value)
        print()

        # Compare each value up to 9th precision
        np_testing.assert_array_almost_equal(expected_value, actual_value, decimal=9)

    def test_calculate_ba(self):
        print(">>>> Test calculate_ba")

        expected_value = np.array([0.028571429, 0.006993007, 0.017782218])
        actual_value = calculate_ba(test_cm, MAIN_DIR_PATH)

        print("-- Expected: ")
        print(expected_value)
        print("-- Actual")
        print(actual_value)
        print()

        # Compare each value up to 9th precision
        np_testing.assert_array_almost_equal(expected_value, actual_value, decimal=9)


if __name__ == '__main__':
    unittest.main()
