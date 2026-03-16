"""CSV telemetry logging for LapEdge."""

import csv
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QObject, QTimer


class TelemetryLogger(QObject):
    """Buffers telemetry frames and flushes to CSV periodically."""

    # CSV columns matching TelemetryFrame fields
    COLUMNS = [
        "timestamp", "lap", "lap_time", "speed", "rpm", "gear",
        "throttle", "brake", "clutch", "steering",
        "fuel_level", "fuel_pct",
        "tire_wear_lf", "tire_wear_rf", "tire_wear_lr", "tire_wear_rr",
        "brake_temp_lf", "brake_temp_rf", "brake_temp_lr", "brake_temp_rr",
        "lat_accel", "lon_accel", "yaw_rate",
        "track_pct", "position", "cars_in_class",
        "on_pit_road", "session_time"
    ]

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config.logging
        self._log_dir = Path(config.logging.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = deque(maxlen=600)  # ~10s at 60hz
        self._file = None
        self._writer = None
        self._file_path = None

        self._flush_timer = QTimer(self)
        self._flush_timer.timeout.connect(self._flush)

    def start_session(self, car_name: str, track_name: str):
        """Open a new CSV file for this session."""
        if not self._config.enabled:
            return

        self._close_file()

        safe_car = car_name.replace(" ", "_").replace("/", "-")[:30]
        safe_track = track_name.replace(" ", "_").replace("/", "-")[:30]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_car}_{safe_track}_{ts}.csv"
        self._file_path = self._log_dir / filename

        self._file = open(self._file_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS,
                                       extrasaction="ignore")
        self._writer.writeheader()

        self._flush_timer.start(self._config.flush_interval_s * 1000)

    def log_frame(self, frame):
        """Buffer a telemetry frame for writing."""
        if not self._config.enabled or self._writer is None:
            return

        row = {
            "timestamp": time.time(),
            "lap": frame.lap,
            "lap_time": frame.lap_time,
            "speed": frame.speed,
            "rpm": frame.rpm,
            "gear": frame.gear,
            "throttle": frame.throttle,
            "brake": frame.brake,
            "clutch": frame.clutch,
            "steering": frame.steering,
            "fuel_level": frame.fuel_level,
            "fuel_pct": frame.fuel_pct,
            "tire_wear_lf": frame.tire_wear_lf,
            "tire_wear_rf": frame.tire_wear_rf,
            "tire_wear_lr": frame.tire_wear_lr,
            "tire_wear_rr": frame.tire_wear_rr,
            "brake_temp_lf": frame.brake_temp_lf,
            "brake_temp_rf": frame.brake_temp_rf,
            "brake_temp_lr": frame.brake_temp_lr,
            "brake_temp_rr": frame.brake_temp_rr,
            "lat_accel": frame.lat_accel,
            "lon_accel": frame.lon_accel,
            "yaw_rate": frame.yaw_rate,
            "track_pct": frame.track_pct,
            "position": frame.position,
            "cars_in_class": frame.cars_in_class,
            "on_pit_road": frame.on_pit_road,
            "session_time": frame.session_time,
        }
        self._buffer.append(row)

    def _flush(self):
        """Write buffered rows to disk."""
        if not self._buffer or self._writer is None:
            return

        rows = list(self._buffer)
        self._buffer.clear()
        self._writer.writerows(rows)
        self._file.flush()

    def stop(self):
        """Flush remaining data and close file."""
        self._flush_timer.stop()
        self._flush()
        self._close_file()

    def _close_file(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
