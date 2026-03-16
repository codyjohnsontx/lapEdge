"""Feature engineering and tire degradation estimation for LapEdge."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import get_car_config, AppConfig
from telemetry import TelemetryFrame, SessionInfo

NUM_FEATURES = 24


@dataclass
class LapSummary:
    """Aggregated per-lap metrics."""
    lap_number: int = 0
    lap_time: float = 0.0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    avg_throttle: float = 0.0
    avg_brake: float = 0.0
    brake_count: int = 0
    avg_steering_magnitude: float = 0.0
    fuel_used: float = 0.0
    fuel_remaining: float = 0.0
    fuel_remaining_pct: float = 0.0
    avg_lat_g: float = 0.0
    max_lat_g: float = 0.0
    avg_lon_g: float = 0.0
    tire_wear_avg: float = 1.0
    # Estimated fields
    estimated_tire_life_pct: float = 1.0
    avg_corner_speed: float = 0.0
    stint_lap: int = 0  # laps since last pit


@dataclass
class RaceContext:
    """Current race state for strategy decisions."""
    current_lap: int = 0
    total_laps: int = 0
    laps_remaining: int = 0
    time_remaining_s: float = 0.0
    stint_length: int = 0
    fuel_remaining_l: float = 0.0
    fuel_per_lap: float = 0.0
    fuel_laps_remaining: float = 0.0
    estimated_tire_life_pct: float = 1.0
    tire_deg_rate: float = 0.0  # pct per lap
    position: int = 0
    cars_in_class: int = 0
    gap_ahead_s: float = 999.0
    gap_behind_s: float = 999.0
    competitor_pit_status: List[bool] = field(default_factory=list)
    session_type: str = ""
    car_name: str = ""
    track_name: str = ""
    lap_time_trend: float = 0.0  # positive = getting slower
    max_tire_life_laps: int = 60


@dataclass
class FeatureVector:
    """24-element normalized numpy array for ML inference."""
    values: np.ndarray = field(default_factory=lambda: np.zeros(NUM_FEATURES))

    def __post_init__(self):
        if len(self.values) != NUM_FEATURES:
            raise ValueError(f"FeatureVector must have {NUM_FEATURES} elements")


class DataProcessor:
    """Stateful processor that accumulates lap history and computes features."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._lap_summaries: List[LapSummary] = []
        self._frame_buffer: List[TelemetryFrame] = []
        self._session_info: Optional[SessionInfo] = None
        self._stint_start_lap: int = 0
        self._last_fuel_level: float = -1.0
        self._fuel_per_lap_history: List[float] = []
        self._competitors: list = []
        self._gap_ahead: float = 999.0
        self._gap_behind: float = 999.0

    def reset(self):
        """Reset all state for a new session."""
        self._lap_summaries.clear()
        self._frame_buffer.clear()
        self._stint_start_lap = 0
        self._last_fuel_level = -1.0
        self._fuel_per_lap_history.clear()
        self._competitors.clear()

    def set_session_info(self, info: SessionInfo):
        self._session_info = info

    def set_competitors(self, competitors: list):
        self._competitors = competitors
        self._compute_gaps()

    def buffer_frame(self, frame: TelemetryFrame):
        """Buffer a frame for per-lap aggregation."""
        self._frame_buffer.append(frame)

    def process_lap(self, frame: TelemetryFrame, lap_time: float):
        """Process a completed lap. Returns (LapSummary, RaceContext, FeatureVector)."""
        summary = self._aggregate_lap(frame, lap_time)
        self._lap_summaries.append(summary)

        # Track fuel usage
        if self._last_fuel_level > 0:
            fuel_used = self._last_fuel_level - frame.fuel_level
            if fuel_used > 0:
                self._fuel_per_lap_history.append(fuel_used)
                summary.fuel_used = fuel_used
        self._last_fuel_level = frame.fuel_level

        # Detect pit stop (reset stint)
        if frame.on_pit_road:
            self._stint_start_lap = frame.lap

        summary.stint_lap = frame.lap - self._stint_start_lap

        # Estimate tire degradation from indirect signals
        summary.estimated_tire_life_pct = self._estimate_tire_life(summary)

        context = self._build_race_context(frame, summary)
        features = self._build_feature_vector(summary, context)

        self._frame_buffer.clear()
        return summary, context, features

    def _aggregate_lap(self, frame: TelemetryFrame, lap_time: float) -> LapSummary:
        """Aggregate buffered frames into a LapSummary."""
        frames = self._frame_buffer if self._frame_buffer else [frame]

        speeds = [f.speed for f in frames]
        throttles = [f.throttle for f in frames]
        brakes = [f.brake for f in frames]
        steerings = [abs(f.steering) for f in frames]
        lat_gs = [abs(f.lat_accel) for f in frames]
        lon_gs = [abs(f.lon_accel) for f in frames]

        # Corner speed: frames where steering > 0.05 rad
        corner_speeds = [f.speed for f in frames if abs(f.steering) > 0.05]
        avg_corner_speed = float(np.mean(corner_speeds)) if corner_speeds else 0.0

        # Brake events: transitions from 0 to >0.1
        brake_count = 0
        prev_brake = 0.0
        for f in frames:
            if f.brake > 0.1 and prev_brake <= 0.1:
                brake_count += 1
            prev_brake = f.brake

        tire_wear_avg = np.mean([
            frame.tire_wear_lf, frame.tire_wear_rf,
            frame.tire_wear_lr, frame.tire_wear_rr
        ])

        return LapSummary(
            lap_number=frame.lap,
            lap_time=lap_time,
            avg_speed=float(np.mean(speeds)),
            max_speed=float(np.max(speeds)),
            avg_throttle=float(np.mean(throttles)),
            avg_brake=float(np.mean(brakes)),
            brake_count=brake_count,
            avg_steering_magnitude=float(np.mean(steerings)),
            fuel_remaining=frame.fuel_level,
            fuel_remaining_pct=frame.fuel_pct,
            avg_lat_g=float(np.mean(lat_gs)),
            max_lat_g=float(np.max(lat_gs)),
            avg_lon_g=float(np.mean(lon_gs)),
            tire_wear_avg=float(tire_wear_avg),
            avg_corner_speed=avg_corner_speed,
        )

    def _estimate_tire_life(self, current: LapSummary) -> float:
        """Estimate tire life from indirect signals (0.0=dead, 1.0=new).

        Since iRacing doesn't expose real-time tire temps for most cars,
        we use 5 indirect degradation signals:
        1. Lap time trend (increasing = degradation)
        2. Brake input trend (increasing = compensating for less grip)
        3. Corner speed trend (decreasing = less mechanical grip)
        4. Throttle application changes (earlier/lighter = less traction)
        5. Lateral G-force reduction (less grip = lower cornering G)
        """
        if len(self._lap_summaries) < 3:
            # Not enough data — assume fresh tires
            return 1.0

        car_config = self._get_car_config()
        stint_lap = current.stint_lap
        max_life = car_config.max_tire_life_laps

        # Baseline degradation from stint length
        base_deg = min(stint_lap / max_life, 1.0)

        # Get recent laps for trend analysis (last 5 laps in stint)
        recent = [s for s in self._lap_summaries[-5:] if s.stint_lap > 0]
        if len(recent) < 2:
            return max(0.0, 1.0 - base_deg)

        # Signal 1: Lap time trend (weight: 0.35)
        lap_times = [s.lap_time for s in recent if s.lap_time > 0]
        lap_time_signal = 0.0
        if len(lap_times) >= 2:
            baseline = min(lap_times)
            current_delta = (current.lap_time - baseline) / baseline if baseline > 0 else 0
            lap_time_signal = min(current_delta * 10, 1.0)  # 10% slower = fully degraded

        # Signal 2: Brake input trend (weight: 0.15)
        brakes = [s.avg_brake for s in recent]
        brake_signal = 0.0
        if len(brakes) >= 2 and brakes[0] > 0:
            brake_increase = (brakes[-1] - brakes[0]) / brakes[0]
            brake_signal = min(max(brake_increase * 5, 0), 1.0)

        # Signal 3: Corner speed trend (weight: 0.20)
        corner_speeds = [s.avg_corner_speed for s in recent if s.avg_corner_speed > 0]
        corner_signal = 0.0
        if len(corner_speeds) >= 2 and corner_speeds[0] > 0:
            speed_drop = (corner_speeds[0] - corner_speeds[-1]) / corner_speeds[0]
            corner_signal = min(max(speed_drop * 8, 0), 1.0)

        # Signal 4: Throttle application (weight: 0.10)
        throttles = [s.avg_throttle for s in recent]
        throttle_signal = 0.0
        if len(throttles) >= 2 and throttles[0] > 0:
            throttle_drop = (throttles[0] - throttles[-1]) / throttles[0]
            throttle_signal = min(max(throttle_drop * 8, 0), 1.0)

        # Signal 5: Lateral G reduction (weight: 0.20)
        lat_gs = [s.avg_lat_g for s in recent if s.avg_lat_g > 0]
        lat_g_signal = 0.0
        if len(lat_gs) >= 2 and lat_gs[0] > 0:
            g_drop = (lat_gs[0] - lat_gs[-1]) / lat_gs[0]
            lat_g_signal = min(max(g_drop * 8, 0), 1.0)

        # Weighted composite
        indirect_deg = (
            0.35 * lap_time_signal +
            0.15 * brake_signal +
            0.20 * corner_signal +
            0.10 * throttle_signal +
            0.20 * lat_g_signal
        )

        # Blend base (stint length) and indirect signals
        total_deg = 0.4 * base_deg + 0.6 * indirect_deg
        return max(0.0, min(1.0, 1.0 - total_deg))

    def _build_race_context(self, frame: TelemetryFrame, summary: LapSummary) -> RaceContext:
        """Build RaceContext from current state."""
        car_config = self._get_car_config()

        avg_fuel = (
            np.mean(self._fuel_per_lap_history[-5:])
            if self._fuel_per_lap_history
            else 0.0
        )
        fuel_laps = frame.fuel_level / avg_fuel if avg_fuel > 0 else 999.0

        # Lap time trend: positive means getting slower
        lap_time_trend = 0.0
        valid_laps = [s for s in self._lap_summaries[-5:] if s.lap_time > 0]
        if len(valid_laps) >= 3:
            times = [s.lap_time for s in valid_laps]
            x = np.arange(len(times))
            coeffs = np.polyfit(x, times, 1)
            lap_time_trend = coeffs[0]

        # Tire deg rate per lap
        tire_deg_rate = 0.0
        if summary.stint_lap > 0:
            tire_deg_rate = (1.0 - summary.estimated_tire_life_pct) / summary.stint_lap

        return RaceContext(
            current_lap=frame.lap,
            total_laps=frame.lap + (frame.session_laps_remain or 0),
            laps_remaining=frame.session_laps_remain or 0,
            time_remaining_s=frame.session_time_remain,
            stint_length=summary.stint_lap,
            fuel_remaining_l=frame.fuel_level,
            fuel_per_lap=float(avg_fuel),
            fuel_laps_remaining=float(fuel_laps),
            estimated_tire_life_pct=summary.estimated_tire_life_pct,
            tire_deg_rate=tire_deg_rate,
            position=frame.position,
            cars_in_class=frame.cars_in_class,
            gap_ahead_s=self._gap_ahead,
            gap_behind_s=self._gap_behind,
            competitor_pit_status=[c.on_pit_road for c in self._competitors],
            session_type=self._session_info.session_type if self._session_info else "",
            car_name=self._session_info.car_name if self._session_info else "",
            track_name=self._session_info.track_name if self._session_info else "",
            lap_time_trend=lap_time_trend,
            max_tire_life_laps=car_config.max_tire_life_laps,
        )

    def _build_feature_vector(self, summary: LapSummary, context: RaceContext) -> FeatureVector:
        """Build normalized 24-feature vector for ML inference.

        Features (all normalized to [0, 1]):
         0: stint_length / max_tire_life
         1: fuel_remaining_pct
         2: fuel_laps_remaining (capped at 50)
         3: estimated_tire_life_pct
         4: tire_deg_rate (capped at 0.1/lap)
         5: lap_time_normalized (vs best)
         6: lap_time_trend (capped)
         7: avg_speed_normalized
         8: avg_throttle
         9: avg_brake
        10: avg_steering_magnitude (capped at 1.0)
        11: avg_corner_speed_normalized
        12: avg_lat_g (capped at 3.0G)
        13: max_lat_g (capped at 5.0G)
        14: avg_lon_g (capped at 3.0G)
        15: position_normalized
        16: gap_ahead_normalized (capped at 30s)
        17: gap_behind_normalized (capped at 30s)
        18: laps_remaining_normalized (capped at 100)
        19: competitors_pitting_pct
        20: brake_count_normalized (capped at 20)
        21: race_progress (current_lap / total_laps)
        22: fuel_per_lap_normalized (capped at 5L)
        23: stint_fuel_efficiency (fuel_used_this_stint relative)
        """
        car_config = self._get_car_config()
        best_lap = min(
            (s.lap_time for s in self._lap_summaries if s.lap_time > 0),
            default=summary.lap_time or 1.0
        )

        total_competitors = len(self._competitors) or 1
        pitting_count = sum(1 for c in self._competitors if c.on_pit_road)

        raw = np.array([
            summary.stint_lap / max(car_config.max_tire_life_laps, 1),
            summary.fuel_remaining_pct,
            min(context.fuel_laps_remaining / 50.0, 1.0),
            context.estimated_tire_life_pct,
            min(context.tire_deg_rate / 0.1, 1.0),
            summary.lap_time / best_lap if best_lap > 0 else 1.0,
            min(max(context.lap_time_trend + 0.5, 0), 1.0),  # center around 0.5
            min(summary.avg_speed / 100.0, 1.0),  # ~100 m/s = 360 kph
            summary.avg_throttle,
            summary.avg_brake,
            min(summary.avg_steering_magnitude, 1.0),
            min(summary.avg_corner_speed / 80.0, 1.0),
            min(summary.avg_lat_g / 3.0, 1.0),
            min(summary.max_lat_g / 5.0, 1.0),
            min(summary.avg_lon_g / 3.0, 1.0),
            context.position / max(context.cars_in_class, 1),
            min(context.gap_ahead_s / 30.0, 1.0),
            min(context.gap_behind_s / 30.0, 1.0),
            min(context.laps_remaining / 100.0, 1.0),
            pitting_count / total_competitors,
            min(summary.brake_count / 20.0, 1.0),
            context.current_lap / max(context.total_laps, 1),
            min(context.fuel_per_lap / 5.0, 1.0),
            min(summary.fuel_used / max(context.fuel_per_lap, 0.1), 1.0)
            if context.fuel_per_lap > 0 else 0.5,
        ], dtype=np.float32)

        # Clamp to [0, 1]
        raw = np.clip(raw, 0.0, 1.0)
        return FeatureVector(values=raw)

    def _compute_gaps(self):
        """Compute gap to car ahead and behind from competitor data."""
        # Simplified — in reality you'd compute from track position deltas
        self._gap_ahead = 999.0
        self._gap_behind = 999.0

        for c in self._competitors:
            if c.gap_ahead_s > 0 and c.gap_ahead_s < self._gap_ahead:
                self._gap_ahead = c.gap_ahead_s
            if c.gap_behind_s > 0 and c.gap_behind_s < self._gap_behind:
                self._gap_behind = c.gap_behind_s

    def _get_car_config(self):
        car_name = self._session_info.car_name if self._session_info else ""
        return get_car_config(self._config, car_name)
