import unittest

import pandas as pd

from src.aggregate import aggregate_metric_rows, paired_bootstrap_delta


class AggregateTests(unittest.TestCase):
    def test_mean_and_sample_sd(self):
        rows = [
            {"experiment": "BASE", "seed": 42, "macro_f1": 0.6,
             "accuracy": 0.8, "weighted_f1": 0.75, "clean_fpr": 0.1, "hate_recall": 0.4},
            {"experiment": "BASE", "seed": 52, "macro_f1": 0.8,
             "accuracy": 0.9, "weighted_f1": 0.85, "clean_fpr": 0.2, "hate_recall": 0.6},
        ]
        result = aggregate_metric_rows(rows)[0]
        self.assertAlmostEqual(result["macro_f1"]["mean"], 0.7)
        self.assertIsNotNone(result["macro_f1"]["sample_sd"])

    def test_paired_bootstrap_alignment(self):
        base = pd.DataFrame({
            "sample_id": [f"x{i}" for i in range(6)],
            "y_true": [0, 0, 1, 1, 2, 2],
            "y_pred": [0, 1, 1, 1, 0, 2],
        })
        proposed = pd.DataFrame({
            "sample_id": list(reversed([f"x{i}" for i in range(6)])),
            "y_true": list(reversed([0, 0, 1, 1, 2, 2])),
            "y_pred": list(reversed([0, 0, 1, 1, 2, 2])),
        })
        result = paired_bootstrap_delta(
            {42: base, 52: base}, {42: proposed, 52: proposed},
            iterations=50, seed=1,
        )
        self.assertGreater(result["observed_delta_macro_f1"], 0)


if __name__ == "__main__":
    unittest.main()
