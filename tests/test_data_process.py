import unittest

import numpy as np

from lib.data_process import split_data_by_ratio, time_add


class DataProcessTests(unittest.TestCase):
    def test_supported_numpy_keeps_matrix_api_used_by_graph_models(self):
        adjacency = np.array([[2.0, 1.0], [1.0, 2.0]])
        degree = adjacency.sum(axis=1)
        inverse_sqrt_degree = np.sqrt(np.mat(np.diag(degree)).I)
        normalized = np.mat(
            np.identity(adjacency.shape[0])
            + inverse_sqrt_degree * adjacency * inverse_sqrt_degree
        )

        np.testing.assert_allclose(
            normalized,
            np.array([[5 / 3, 1 / 3], [1 / 3, 5 / 3]]),
        )

    def test_non_midnight_start_keeps_time_features_aligned_to_rows(self):
        data = np.ones((60, 1), dtype=np.float32)

        day, week, _ = time_add(
            data, week_start=1, interval=30, day_start=18 * 60
        )

        self.assertTrue(np.all(day[:, 0] > 0))
        self.assertEqual(day[0, 0], 18 * 60 + 30)
        self.assertEqual(day[11, 0], 24 * 60)
        self.assertEqual(day[12, 0], 30)
        self.assertEqual(week[0, 0], 1)
        self.assertEqual(week[12, 0], 2)

    def test_zero_test_ratio_retains_requested_validation_tail(self):
        data = np.arange(100)

        train, validation, test = split_data_by_ratio(
            data, val_ratio=0.2, test_ratio=0
        )

        np.testing.assert_array_equal(train, np.arange(80))
        np.testing.assert_array_equal(validation, np.arange(80, 100))
        self.assertEqual(len(test), 0)

    def test_zero_validation_and_test_ratios_keep_all_training_data(self):
        data = np.arange(100)

        train, validation, test = split_data_by_ratio(
            data, val_ratio=0, test_ratio=0
        )

        np.testing.assert_array_equal(train, data)
        self.assertEqual(len(validation), 0)
        self.assertEqual(len(test), 0)


if __name__ == "__main__":
    unittest.main()
