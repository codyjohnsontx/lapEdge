"""Configuration management for LapEdge."""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# Alert mode constants
PASSIVE = "passive"
ON_DEMAND = "on_demand"
PERIODIC = "periodic"

ALERT_MODES = [PASSIVE, ON_DEMAND, PERIODIC]

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default_config.json"
USER_CONFIG_PATH = Path.home() / ".lapedge" / "config.json"


@dataclass
class OverlayConfig:
    x: int = 100
    y: int = 100
    width: int = 420
    height: int = 320
    opacity: float = 0.85
    font_size: int = 14
    alert_mode: str = PERIODIC
    periodic_interval_laps: int = 5


@dataclass
class TelemetryConfig:
    capture_rate_hz: int = 60
    ui_update_rate_hz: int = 4
    session_info_interval_s: int = 5


@dataclass
class LoggingConfig:
    enabled: bool = True
    flush_interval_s: int = 5
    log_dir: str = "logs"


@dataclass
class ModelConfig:
    enabled: bool = True
    model_dir: str = "models"


@dataclass
class HotkeyConfig:
    toggle_overlay: str = "<ctrl>+<shift>+l"
    cycle_mode: str = "<ctrl>+<shift>+m"
    request_update: str = "<ctrl>+<shift>+r"


@dataclass
class VoiceConfig:
    enabled: bool = True
    volume: float = 0.9
    rate: int = 175
    min_repeat_interval_s: float = 20.0
    speak_advisory: bool = True
    speak_info: bool = False
    voice_id: str = ""


@dataclass
class CarOverride:
    max_tire_life_laps: int = 60
    fuel_tank_capacity_l: float = 80.0
    compounds: list = field(default_factory=lambda: ["hard"])


@dataclass
class AppConfig:
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    car_overrides: dict = field(default_factory=dict)
    defaults: CarOverride = field(default_factory=CarOverride)
    voice: VoiceConfig = field(default_factory=VoiceConfig)


def _dict_to_config(data: dict) -> AppConfig:
    """Convert a nested dict to AppConfig."""
    config = AppConfig()

    if "overlay" in data:
        config.overlay = OverlayConfig(**data["overlay"])
    if "telemetry" in data:
        config.telemetry = TelemetryConfig(**data["telemetry"])
    if "logging" in data:
        config.logging = LoggingConfig(**data["logging"])
    if "model" in data:
        config.model = ModelConfig(**data["model"])
    if "hotkeys" in data:
        config.hotkeys = HotkeyConfig(**data["hotkeys"])
    if "defaults" in data:
        config.defaults = CarOverride(**data["defaults"])
    if "car_overrides" in data:
        config.car_overrides = {
            k: CarOverride(**v) for k, v in data["car_overrides"].items()
        }
    if "voice" in data:
        config.voice = VoiceConfig(**data["voice"])

    return config


def _config_to_dict(config: AppConfig) -> dict:
    """Convert AppConfig to a serializable dict."""
    d = asdict(config)
    # car_overrides values are already dicts from asdict
    return d


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load config from JSON file. Falls back to defaults if file missing."""
    # Start with defaults
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            base = json.load(f)
    else:
        base = {}

    # Layer user config on top
    config_path = Path(path) if path else USER_CONFIG_PATH
    if config_path.exists():
        with open(config_path, "r") as f:
            user = json.load(f)
        # Shallow merge per section
        for key, val in user.items():
            if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                base[key].update(val)
            else:
                base[key] = val

    return _dict_to_config(base)


def save_config(config: AppConfig, path: Optional[str] = None) -> None:
    """Save config to JSON file."""
    config_path = Path(path) if path else USER_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(_config_to_dict(config), f, indent=4)


def get_car_config(config: AppConfig, car_name: str) -> CarOverride:
    """Get car-specific config, falling back to defaults."""
    key = car_name.lower().strip()
    return config.car_overrides.get(key, config.defaults)
