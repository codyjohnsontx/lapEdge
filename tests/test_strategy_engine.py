"""Tests for strategy_engine.py — verify each rule fires correctly."""

import unittest

from data_processor import RaceContext
from strategy_engine import StrategyEngine, Urgency, Recommendation
from model import ModelPrediction


class TestStrategyRules(unittest.TestCase):

    def setUp(self):
        self.engine = StrategyEngine()

    def _base_context(self, **overrides) -> RaceContext:
        """Create a baseline RaceContext with overrides."""
        defaults = dict(
            current_lap=15, total_laps=50, laps_remaining=35,
            time_remaining_s=3600, stint_length=12,
            fuel_remaining_l=30.0, fuel_per_lap=3.0,
            fuel_laps_remaining=10.0,
            estimated_tire_life_pct=0.6, tire_deg_rate=0.02,
            position=5, cars_in_class=20,
            gap_ahead_s=2.5, gap_behind_s=4.0,
            competitor_pit_status=[],
            session_type="Race", car_name="test car",
            track_name="test track", lap_time_trend=0.0,
            max_tire_life_laps=60,
        )
        defaults.update(overrides)
        return RaceContext(**defaults)

    def test_rule1_fuel_critical(self):
        """Fuel < 2 laps → CRITICAL pit now."""
        ctx = self._base_context(fuel_laps_remaining=1.5, fuel_per_lap=3.0)
        rec = self.engine.evaluate(ctx)
        self.assertEqual(rec.urgency, Urgency.CRITICAL)
        self.assertTrue(rec.pit_this_lap)
        self.assertIn("FUEL CRITICAL", rec.message)

    def test_rule2_fuel_warning(self):
        """Fuel < 4 laps → ADVISORY."""
        ctx = self._base_context(fuel_laps_remaining=3.0, fuel_per_lap=3.0)
        rec = self.engine.evaluate(ctx)
        self.assertEqual(rec.urgency, Urgency.ADVISORY)
        self.assertIn("Fuel warning", rec.message)

    def test_rule3_severe_tire_deg(self):
        """Tire life < 10% → ADVISORY."""
        ctx = self._base_context(
            estimated_tire_life_pct=0.05,
            fuel_laps_remaining=20.0,  # fuel is fine
        )
        rec = self.engine.evaluate(ctx)
        self.assertEqual(rec.urgency, Urgency.ADVISORY)
        self.assertIn("Tire life critical", rec.message)

    def test_rule4_undercut(self):
        """Gap ahead < 3s + worn tires → undercut opportunity."""
        ctx = self._base_context(
            gap_ahead_s=2.0,
            estimated_tire_life_pct=0.40,
            stint_length=15,
            fuel_laps_remaining=20.0,
        )
        rec = self.engine.evaluate(ctx)
        self.assertEqual(rec.urgency, Urgency.ADVISORY)
        self.assertIn("Undercut", rec.message)

    def test_rule5_overcut(self):
        """Competitors pitting + our tires OK → overcut."""
        ctx = self._base_context(
            competitor_pit_status=[True, True, False, False],
            estimated_tire_life_pct=0.50,
            gap_ahead_s=5.0,  # No undercut possible
            fuel_laps_remaining=20.0,
        )
        rec = self.engine.evaluate(ctx)
        self.assertEqual(rec.urgency, Urgency.INFO)
        self.assertIn("Overcut", rec.message)

    def test_rule6_safe_pit_window(self):
        """Large gap behind + worn tires → safe pit window."""
        ctx = self._base_context(
            gap_behind_s=30.0,
            estimated_tire_life_pct=0.40,
            stint_length=15,
            gap_ahead_s=10.0,  # No undercut
            competitor_pit_status=[],  # No overcut
            fuel_laps_remaining=20.0,
        )
        rec = self.engine.evaluate(ctx)
        self.assertEqual(rec.urgency, Urgency.INFO)
        self.assertIn("Safe pit window", rec.message)

    def test_rule7_approaching_optimal(self):
        """Tire deg approaching 20% life → advisory with lap count."""
        ctx = self._base_context(
            estimated_tire_life_pct=0.30,
            tire_deg_rate=0.03,
            gap_behind_s=5.0,  # No safe window
            gap_ahead_s=10.0,  # No undercut
            competitor_pit_status=[],
            fuel_laps_remaining=20.0,
        )
        rec = self.engine.evaluate(ctx)
        # Should be ADVISORY with "Optimal pit window in ~N laps"
        self.assertEqual(rec.urgency, Urgency.ADVISORY)
        self.assertIn("Optimal pit window", rec.message)

    def test_rule8_default_stay_out(self):
        """No issues → INFO stay out."""
        ctx = self._base_context(
            fuel_laps_remaining=25.0,
            estimated_tire_life_pct=0.90,
            gap_ahead_s=10.0,
            gap_behind_s=5.0,
            competitor_pit_status=[],
            tire_deg_rate=0.005,
            stint_length=3,
        )
        rec = self.engine.evaluate(ctx)
        self.assertEqual(rec.urgency, Urgency.INFO)
        self.assertIn("Stay out", rec.message)

    def test_fuel_critical_highest_priority(self):
        """Fuel critical should override tire degradation."""
        ctx = self._base_context(
            fuel_laps_remaining=1.0,
            fuel_per_lap=3.0,
            estimated_tire_life_pct=0.05,  # Also critical tires
        )
        rec = self.engine.evaluate(ctx)
        self.assertEqual(rec.urgency, Urgency.CRITICAL)
        self.assertIn("FUEL CRITICAL", rec.message)


class TestMLStrategy(unittest.TestCase):

    def setUp(self):
        self.engine = StrategyEngine()

    def test_high_confidence_pit_now(self):
        """ML says pit now with high confidence → CRITICAL."""
        ctx = RaceContext(
            current_lap=20, total_laps=50, fuel_laps_remaining=15.0,
            estimated_tire_life_pct=0.4, fuel_per_lap=3.0,
        )
        pred = ModelPrediction(
            pit_lap_offset=0, compound="medium",
            confidence=0.9, model_type="tf",
        )
        rec = self.engine.evaluate(ctx, pred)
        self.assertEqual(rec.urgency, Urgency.CRITICAL)
        self.assertTrue(rec.pit_this_lap)
        self.assertIn("ML", rec.message)

    def test_low_confidence_falls_back_to_rules(self):
        """Low confidence ML prediction → fall back to rules."""
        ctx = RaceContext(
            current_lap=20, total_laps=50,
            fuel_laps_remaining=25.0, fuel_per_lap=3.0,
            estimated_tire_life_pct=0.90,
            gap_ahead_s=10.0, gap_behind_s=5.0,
            competitor_pit_status=[], tire_deg_rate=0.005,
            stint_length=3,
        )
        pred = ModelPrediction(
            pit_lap_offset=5, compound="hard",
            confidence=0.3,  # Low confidence
            model_type="tf",
        )
        rec = self.engine.evaluate(ctx, pred)
        # Should fall back to rule-based (confidence < 0.7)
        self.assertNotIn("ML", rec.message)


if __name__ == "__main__":
    unittest.main()
