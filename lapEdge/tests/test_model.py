"""Tests for model.py — fallback behavior when no weights exist."""

import unittest

import numpy as np

from model import PitStrategyModel, ModelPrediction
from data_processor import NUM_FEATURES


class TestModelFallback(unittest.TestCase):

    def test_load_returns_false_no_weights(self):
        """Model load should return False when no weight files exist."""
        model = PitStrategyModel(model_dir="/tmp/nonexistent_model_dir_xyz")
        result = model.load("some car")
        self.assertFalse(result)
        self.assertFalse(model.is_loaded)

    def test_predict_returns_none_when_not_loaded(self):
        """Predict should return None when model isn't loaded."""
        model = PitStrategyModel(model_dir="/tmp/nonexistent_model_dir_xyz")
        features = np.random.rand(NUM_FEATURES).astype(np.float32)
        result = model.predict(features)
        self.assertIsNone(result)

    def test_predict_rejects_wrong_shape(self):
        """Predict should return None for wrong-shaped input."""
        model = PitStrategyModel()
        # Even if somehow loaded, wrong shape should fail
        features = np.random.rand(10).astype(np.float32)
        result = model.predict(features)
        self.assertIsNone(result)

    def test_model_prediction_dataclass(self):
        """ModelPrediction should have correct defaults."""
        pred = ModelPrediction()
        self.assertEqual(pred.pit_lap_offset, 0)
        self.assertEqual(pred.compound, "hard")
        self.assertEqual(pred.confidence, 0.0)
        self.assertEqual(pred.model_type, "none")


if __name__ == "__main__":
    unittest.main()
