import unittest

import numpy as np

from src.calibration import apply_class_biases, fit_macro_f1_biases, macro_f1_score


class CalibrationTests(unittest.TestCase):
    def test_validation_bias_improves_underpredicted_minority(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 2, 2])
        logits = np.array([
            [3.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 0.2, 0.0], [1.2, 0.5, 0.0],
            [1.0, 0.8, 0.0], [0.9, 0.7, 0.1], [0.8, 0.0, 0.7], [0.7, 0.0, 0.6],
        ])
        fitted = fit_macro_f1_biases(logits, labels, radius=0.6, step=0.1)
        adjusted = apply_class_biases(logits, fitted["biases"])
        self.assertGreater(fitted["calibrated_macro_f1"], fitted["raw_macro_f1"])
        self.assertAlmostEqual(fitted["biases"][0], 0.0)
        self.assertAlmostEqual(
            macro_f1_score(labels, adjusted.argmax(axis=-1)),
            fitted["calibrated_macro_f1"],
        )

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_class_biases(np.zeros((2, 2)), [0, 0, 0])
        with self.assertRaises(ValueError):
            fit_macro_f1_biases(np.zeros((2, 3)), [0])


if __name__ == "__main__":
    unittest.main()
