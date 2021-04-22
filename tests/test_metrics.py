import unittest
import torch
from uninas.training.metrics.accuracy import AccuracyMetric
from uninas.builder import Builder


class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        """
        """
        Builder()

        # example classification output, target value
        values = [
            ((0, 0, 0, 1, 2), 4),  # predicts: 4, 3 -> T, T
            ((2, 0, 1, 0, 0), 1),  # predicts: 0, 2 -> F, F
            ((0, 1, 3, 0, 0), 2),  # predicts: 2, 1 -> T, T
            ((1, 2, 4, 0, 0), 1),  # predicts: 2, 1 -> F, T
            ((0, 1, 0, 2, 0), 3),  # predicts: 3, 1 -> T, T
        ]
        outputs = torch.cat([torch.tensor([v[0]], dtype=torch.float32) for v in values], dim=0)
        targets = torch.cat([torch.tensor([v[1]], dtype=torch.float32) for v in values], dim=0)

        t1, t2 = AccuracyMetric.accuracy(outputs, targets, top_k=(1, 2))
        assert t1 == 0.6
        assert t2 == 0.8


if __name__ == '__main__':
    unittest.main()
