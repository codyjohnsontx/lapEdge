"""Tests for data_processor.py — feature vectors, fuel calc, tire degradation."""

import unittest

import numpy as np

from config import AppConfig, CarOverride
from data_processor import DataProcessor, LapSummary, NUM_FEATURES
from telemetry import TelemetryFrame, SessionInfo


def make_frame(**kwargs) -> TelemetryFrame:
    """Create a TelemetryFrame with sensible defaults."""
    defaults = dict(
        lap=5, lap_time=30.0, last_lap_time=90.0, best_lap_time=88.0,
        speed=50.0, rpm=6000, gear=4, throttle=0.7, brake=0.1,
        clutch=0.0, steering=0.1, fuel_level=30.0, fuel_pct=0.5,
        tire_wear_lf=0.8, tire_wear_rf=0.8, tire_wear_lr=0.85, tire_wear_rr=0.85,
        lat_accel=1.2, lon_accel=0.5, yaw_rate=0.3, track_pct=0.5,
        position=3, cars_in_class=20, on_pit_road=False,
        session_time=600.0, session_time_remain=1800.0, session_laps_remain=20,
    )
    defaults.update(kwargs)
    return TelemetryFrame(**defaults)


class TestFeatureVector(unittest.TestCase):

    def test_feature_vector_shape(self):
        """Feature vector must always have exactly 24 elements."""
        config = AppConfig()
        dp = DataProcessor(config)
        dp.set_session_info(SessionInfo(car_name="test car"))

        frame = make_frame(lap=2, fuel_level=35.0)
        dp.buffer_frame(frame)

        # Need at least one prior lap for fuel calculation
        dp._last_fuel_level = 37.0
        _, _, fv = dp.process_lap(frame, 90.0)
        self.assertEqual(fv.values.shape, (NUM_FEATURES,))
        self.assertEqual(len(fv.values), 24)

    def test_feature_values_in_range(self):
        """All features must be in [0, 1] range."""
        config = AppConfig()
        dp = DataProcessor(config)
        dp.set_session_info(SessionInfo(car_name="test car"))

        frame = make_frame(lap=2, fuel_level=33.0)
        dp.buffer_frame(frame)
        dp._last_fuel_level = 35.0
        _, _, fv = dp.process_lap(frame, 91.0)

        for i, val in enumerate(fv.values):
            self.assertGreaterEqual(val, 0.0, f"Feature {i} below 0: {val}")
            self.assertLessEqual(val, 1.0, f"Feature {i} above 1: {val}")


class TestFuelCalculation(unittest.TestCase):

    def test_fuel_per_lap(self):
        """Fuel per lap should be correctly calculated from level delta."""
        config = AppConfig()
        dp = DataProcessor(config)
        dp.set_session_info(SessionInfo(car_name="test car"))

        # Simulate 3 laps with decreasing fuel
        for lap_num in range(1, 4):
            frame = make_frame(lap=lap_num, fuel_level=40.0 - lap_num * 2.5)
            dp.buffer_frame(frame)
            dp.process_lap(frame, 90.0)

        # Should have 2 fuel measurements (gap between lap 1→2 and 2→3)
        self.assertEqual(len(dp._fuel_per_lap_history), 2)
        self.assertAlmostEqual(dp._fuel_per_lap_history[0], 2.5, places=1)

    def test_fuel_laps_remaining(self):
        """Fuel laps remaining should be fuel_level / fuel_per_lap."""
        config = AppConfig()
        dp = DataProcessor(config)
        dp.set_session_info(SessionInfo(car_name="test car"))

        # Prime fuel history
        dp._fuel_per_lap_history = [3.0, 3.0, 3.0]
        dp._last_fuel_level = 15.0

        frame = make_frame(lap=5, fuel_level=12.0)
        dp.buffer_frame(frame)
        _, context, _ = dp.process_lap(frame, 90.0)

        # 12.0L / 3.0L per lap = 4.0 laps
        self.assertAlmostEqual(context.fuel_laps_remaining, 4.0, places=1)


class TestTireDegradation(unittest.TestCase):

    def test_fresh_tires_high_life(self):
        """Early in stint with good lap times should show high tire life."""
        config = AppConfig()
        dp = DataProcessor(config)
        dp.set_session_info(SessionInfo(car_name="mx5 cup"))
        config.car_overrides = {
            "mx5 cup": CarOverride(max_tire_life_laps=80)
        }

        # Simulate 3 consistent laps (early stint)
        for i in range(1, 4):
            frame = make_frame(
                lap=i, fuel_level=40.0 - i,
                speed=50.0, steering=0.1,
            )
            dp.buffer_frame(frame)
            summary, _, _ = dp.process_lap(frame, 90.0 + 0.1 * i)

        self.assertGreater(summary.estimated_tire_life_pct, 0.8)

    def test_degrading_laps_lower_life(self):
        """Worsening lap times should produce lower tire life estimate."""
        config = AppConfig()
        config.car_overrides = {
            "test car": CarOverride(max_tire_life_laps=30)
        }
        dp = DataProcessor(config)
        dp.set_session_info(SessionInfo(car_name="test car"))

        # Simulate laps getting progressively slower
        for i in range(1, 8):
            frame = make_frame(
                lap=i, fuel_level=40.0 - i,
                speed=50.0 - i * 2,       # slowing down
                throttle=0.7 - i * 0.02,  # less throttle
                brake=0.1 + i * 0.02,     # more braking
                lat_accel=1.5 - i * 0.1,  # less lateral G
                steering=0.15,
            )
            dp.buffer_frame(frame)
            summary, _, _ = dp.process_lap(frame, 90.0 + i * 1.5)

        # After 7 degrading laps, tire life should be noticeably lower
        self.assertLess(summary.estimated_tire_life_pct, 0.7)


if __name__ == "__main__":
    unittest.main()
