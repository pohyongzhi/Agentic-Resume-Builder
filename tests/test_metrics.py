import unittest
from unittest.mock import patch

import numpy as np

from eval import metrics


class ExactMatchNLI:
    """Entails only an exact evidence/claim match; long mixed chunks are neutral."""

    def predict(self, pairs):
        return np.asarray([
            [-3.0, 5.0, -2.0] if premise == claim else [-2.0, -3.0, 5.0]
            for premise, claim in pairs
        ])


class FaithfulnessTests(unittest.TestCase):
    def test_source_claims_are_used_as_evidence_units(self):
        source = "- Built weekly sales dashboards in Excel and Power BI for the regional team."
        with patch.object(metrics, "_nli", ExactMatchNLI()):
            result = metrics.faithfulness(source, source)
        self.assertEqual(result["faithfulness"], 1.0)
        self.assertEqual(result["unsupported"], [])


if __name__ == "__main__":
    unittest.main()
