import unittest

import numpy as np
import torch

from lib.metrics import All_Metrics, CORR_np, CORR_torch


class CorrelationMetricTests(unittest.TestCase):
    def test_torch_perfect_predictions_have_unit_correlation(self):
        true = torch.arange(1, 17, dtype=torch.float32).reshape(2, 4, 2, 1)

        self.assertAlmostEqual(CORR_torch(true, true).item(), 1.0, places=6)

    def test_numpy_two_dimensional_inputs_are_supported(self):
        true = np.array(
            [[1.0, 10.0], [2.0, 12.0], [3.0, 14.0], [4.0, 16.0]],
            dtype=np.float32,
        )
        pred = np.array(
            [[1.0, 14.0], [3.0, 10.0], [2.0, 16.0], [4.0, 12.0]],
            dtype=np.float32,
        )
        expected = np.mean(
            [np.corrcoef(pred[:, node], true[:, node])[0, 1] for node in range(true.shape[1])]
        )

        self.assertAlmostEqual(CORR_np(pred, true), expected, places=6)

    def test_numpy_four_dimensional_correlation_is_per_node(self):
        true = np.empty((2, 4, 2, 1), dtype=np.float32)
        true[:, :, 0, 0] = np.arange(8, dtype=np.float32).reshape(2, 4)
        node_one = np.arange(8, dtype=np.float32).reshape(2, 4)
        true[:, :, 1, 0] = node_one + 100.0
        pred = np.empty_like(true)
        pred[:, :, 0, 0] = true[:, :, 0, 0]
        pred[:, :, 1, 0] = -node_one

        self.assertAlmostEqual(CORR_np(pred, true), 0.0, places=6)

    def test_numpy_all_metrics_returns_mask_counts(self):
        true = np.arange(1, 9, dtype=np.float32).reshape(2, 4, 1, 1)
        pred = true + 0.5

        metrics = All_Metrics(pred, true, 0, 0)

        self.assertEqual(metrics[5:], (8, 8, 8, 8))


if __name__ == "__main__":
    unittest.main()
