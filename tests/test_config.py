"""Tests for config.py — round-trip serialization, car overrides."""

import json
import os
import tempfile
import unittest

from config import (
    AppConfig, OverlayConfig, CarOverride, load_config, save_config,
    get_car_config, PASSIVE, PERIODIC,
)


class TestConfig(unittest.TestCase):

    def test_default_config_loads(self):
        """Default config file should load without errors."""
        config = load_config()
        self.assertIsInstance(config, AppConfig)
        self.assertIsInstance(config.overlay, OverlayConfig)
        self.assertGreater(config.overlay.width, 0)

    def test_round_trip_serialization(self):
        """Config saved and loaded should be equivalent."""
        config = AppConfig()
        config.overlay.opacity = 0.5
        config.overlay.alert_mode = PASSIVE

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_config(config, path)
            loaded = load_config(path)
            self.assertEqual(loaded.overlay.opacity, 0.5)
            self.assertEqual(loaded.overlay.alert_mode, PASSIVE)
        finally:
            os.unlink(path)

    def test_car_override_found(self):
        """Car-specific config should be returned when available."""
        config = AppConfig()
        config.car_overrides = {
            "mx5 cup": CarOverride(
                max_tire_life_laps=80,
                fuel_tank_capacity_l=45.0,
                compounds=["hard"],
            )
        }
        car_cfg = get_car_config(config, "MX5 Cup")
        self.assertEqual(car_cfg.max_tire_life_laps, 80)
        self.assertEqual(car_cfg.fuel_tank_capacity_l, 45.0)

    def test_car_override_fallback(self):
        """Unknown car should fall back to defaults."""
        config = AppConfig()
        config.defaults = CarOverride(max_tire_life_laps=60)
        car_cfg = get_car_config(config, "Unknown Car 3000")
        self.assertEqual(car_cfg.max_tire_life_laps, 60)

    def test_missing_file_returns_defaults(self):
        """Loading from a nonexistent path should return default config."""
        config = load_config("/tmp/nonexistent_lapedge_config_xyz.json")
        self.assertIsInstance(config, AppConfig)
        self.assertGreater(config.overlay.width, 0)


if __name__ == "__main__":
    unittest.main()
