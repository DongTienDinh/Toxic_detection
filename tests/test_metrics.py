import unittest

import numpy as np

from src.metrics import detailed_metrics, prediction_rows, stable_softmax


class MetricsTests(unittest.TestCase):
    def test_known_confusion_and_zero_division(self):
        result = detailed_metrics([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 1, 1])
        self.assertAlmostEqual(result["accuracy"], 2 / 6)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [1, 1, 0], [0, 2, 0]])
        self.assertEqual(result["per_class"]["HATE"]["f1"], 0.0)
        self.assertEqual(result["clean_fpr"], 0.5)

    def test_softmax_and_prediction_schema(self):
        logits = np.array([[3.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
        probabilities = stable_softmax(logits)
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
        sidecar = [
            {"sample_id": "test_00000000", "labels": 0},
            {"sample_id": "test_00000001", "labels": 2},
        ]
        rows = prediction_rows(
            sidecar=sidecar,
            logits=logits,
            experiment_id="BASE",
            seed=42,
            checkpoint_hash="abc",
            protocol_version="v2",
        )
        self.assertEqual(rows[0]["y_pred"], 0)
        self.assertAlmostEqual(rows[1]["p_clean"], 1 / 3)

    def test_invalid_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            detailed_metrics([0, 3], [0, 1])


if __name__ == "__main__":
    unittest.main()
