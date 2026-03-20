"""iRacing telemetry capture via pyirsdk shared memory."""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from PyQt5.QtCore import QObject, pyqtSignal, QTimer

try:
    import irsdk
except ImportError:
    irsdk = None


class SessionFlags:
    """iRacing SDK SessionFlags bitmask constants."""
    CHECKERED      = 0x0001
    WHITE          = 0x0002
    GREEN          = 0x0004
    YELLOW         = 0x0008
    RED            = 0x0010
    BLUE           = 0x0020
    DEBRIS         = 0x0040
    CROSSED        = 0x0080
    YELLOW_WAVING  = 0x0100
    ONE_LAP_TO_GO  = 0x0200
    GREEN_HELD     = 0x0400
    TEN_TO_GO      = 0x0800
    FIVE_TO_GO     = 0x1000
    RANDOM_WAVING  = 0x2000
    CAUTION        = 0x4000
    CAUTION_WAVING = 0x8000
    BLACK          = 0x10000
    DISQUALIFY     = 0x20000
    SERVICEABLE    = 0x40000
    FURLED         = 0x80000
    REPAIR         = 0x100000
    START_HIDDEN   = 0x10000000
    START_READY    = 0x20000000
    START_SET      = 0x40000000
    START_GO       = 0x80000000


@dataclass
class TelemetryFrame:
    """Flat struct of all captured telemetry variables per tick."""
    # Lap info
    lap: int = 0
    lap_time: float = 0.0
    last_lap_time: float = 0.0
    best_lap_time: float = 0.0

    # Vehicle dynamics
    speed: float = 0.0        # m/s
    rpm: float = 0.0
    gear: int = 0
    throttle: float = 0.0     # 0-1
    brake: float = 0.0        # 0-1
    clutch: float = 0.0       # 0-1
    steering: float = 0.0     # radians

    # Fuel
    fuel_level: float = 0.0   # liters
    fuel_pct: float = 0.0     # 0-1

    # Tire wear (0-1, 1=new, may be 0 if unavailable)
    tire_wear_lf: float = 1.0
    tire_wear_rf: float = 1.0
    tire_wear_lr: float = 1.0
    tire_wear_rr: float = 1.0

    # Brake temps (celsius)
    brake_temp_lf: float = 0.0
    brake_temp_rf: float = 0.0
    brake_temp_lr: float = 0.0
    brake_temp_rr: float = 0.0

    # G-forces
    lat_accel: float = 0.0    # lateral G
    lon_accel: float = 0.0    # longitudinal G
    yaw_rate: float = 0.0     # rad/s

    # Position
    track_pct: float = 0.0    # 0-1 position on track
    position: int = 0         # race position
    cars_in_class: int = 0

    # Pit
    on_pit_road: bool = False

    # Session
    session_time: float = 0.0
    session_time_remain: float = 0.0
    session_laps_remain: int = 0
    session_flags: int = 0


@dataclass
class SessionInfo:
    """Parsed session metadata from iRacing YAML."""
    car_name: str = ""
    car_id: int = 0
    track_name: str = ""
    track_id: int = 0
    track_length_km: float = 0.0
    session_type: str = ""      # Practice, Race, Qualify
    pit_speed_limit_mps: float = 0.0
    max_fuel_pct: float = 1.0
    driver_idx: int = 0


@dataclass
class CompetitorInfo:
    """Per-competitor state for strategy decisions."""
    car_idx: int = 0
    position: int = 0
    class_position: int = 0
    last_lap_time: float = 0.0
    best_lap_time: float = 0.0
    gap_ahead_s: float = 0.0
    gap_behind_s: float = 0.0
    laps_completed: int = 0
    on_pit_road: bool = False
    pitting_this_lap: bool = False


class TelemetryWorker(QObject):
    """Captures iRacing telemetry at 60hz in a dedicated QThread."""

    frame_ready = pyqtSignal(object)          # TelemetryFrame
    lap_completed = pyqtSignal(object, float)  # TelemetryFrame, lap_time
    session_info_updated = pyqtSignal(object)  # SessionInfo
    competitors_updated = pyqtSignal(list)      # List[CompetitorInfo]
    connection_status = pyqtSignal(bool)        # connected/disconnected

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config.telemetry
        self._running = False
        self._connected = False
        self._ir = None
        self._last_lap = -1
        self._last_session_parse = 0.0
        self._session_info = SessionInfo()
        self._timer = None

    def start(self):
        """Begin the telemetry capture loop."""
        if irsdk is None:
            print("[telemetry] pyirsdk not available — running in stub mode")
            self.connection_status.emit(False)
            return

        self._ir = irsdk.IRSDK()
        self._running = True

        interval_ms = max(1, int(1000 / self._config.capture_rate_hz))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(interval_ms)

    def stop(self):
        """Stop the capture loop."""
        self._running = False
        if self._timer is not None:
            self._timer.stop()
        if self._ir is not None and self._connected:
            self._ir.shutdown()
            self._connected = False

    def _tick(self):
        """Single capture iteration."""
        if not self._running:
            return

        # Connection management
        if not self._connected:
            if self._ir.startup():
                self._connected = True
                self.connection_status.emit(True)
                self._parse_session_info()
            else:
                return
        elif not self._ir.is_connected:
            self._connected = False
            self.connection_status.emit(False)
            self._last_lap = -1
            return

        # Freeze telemetry snapshot
        self._ir.freeze_var_buffer_latest()

        frame = self._capture_frame()
        self.frame_ready.emit(frame)

        # Lap detection
        if self._last_lap >= 0 and frame.lap > self._last_lap:
            lap_time = frame.last_lap_time
            self.lap_completed.emit(frame, lap_time)
        self._last_lap = frame.lap

        # Periodic session info re-parse
        now = time.time()
        if now - self._last_session_parse > self._config.session_info_interval_s:
            self._parse_session_info()
            self._parse_competitors()
            self._last_session_parse = now

    def _capture_frame(self) -> TelemetryFrame:
        """Read all telemetry variables into a TelemetryFrame."""
        ir = self._ir
        return TelemetryFrame(
            lap=ir["Lap"] or 0,
            lap_time=ir["LapCurrentLapTime"] or 0.0,
            last_lap_time=ir["LapLastLapTime"] or 0.0,
            best_lap_time=ir["LapBestLapTime"] or 0.0,
            speed=ir["Speed"] or 0.0,
            rpm=ir["RPM"] or 0.0,
            gear=ir["Gear"] or 0,
            throttle=ir["Throttle"] or 0.0,
            brake=ir["Brake"] or 0.0,
            clutch=ir["Clutch"] or 0.0,
            steering=ir["SteeringWheelAngle"] or 0.0,
            fuel_level=ir["FuelLevel"] or 0.0,
            fuel_pct=ir["FuelLevelPct"] or 0.0,
            tire_wear_lf=ir["LFwearL"] or 1.0 if ir["LFwearL"] is not None else 1.0,
            tire_wear_rf=ir["RFwearL"] or 1.0 if ir["RFwearL"] is not None else 1.0,
            tire_wear_lr=ir["LRwearL"] or 1.0 if ir["LRwearL"] is not None else 1.0,
            tire_wear_rr=ir["RRwearL"] or 1.0 if ir["RRwearL"] is not None else 1.0,
            brake_temp_lf=ir["LFbrakeLinePress"] or 0.0,
            brake_temp_rf=ir["RFbrakeLinePress"] or 0.0,
            brake_temp_lr=ir["LRbrakeLinePress"] or 0.0,
            brake_temp_rr=ir["RRbrakeLinePress"] or 0.0,
            lat_accel=ir["LatAccel"] or 0.0,
            lon_accel=ir["LongAccel"] or 0.0,
            yaw_rate=ir["YawRate"] or 0.0,
            track_pct=ir["LapDistPct"] or 0.0,
            position=ir["PlayerCarClassPosition"] or 0,
            cars_in_class=ir["PlayerCarClassPosition"] or 0,
            on_pit_road=bool(ir["OnPitRoad"]),
            session_time=ir["SessionTime"] or 0.0,
            session_time_remain=ir["SessionTimeRemain"] or 0.0,
            session_laps_remain=ir["SessionLapsRemainEx"] or 0,
            session_flags=ir["SessionFlags"] or 0,
        )

    def _parse_session_info(self):
        """Parse the YAML session info blob."""
        if self._ir is None or not self._connected:
            return

        try:
            si = self._ir["SessionInfo"]
            wi = self._ir["WeekendInfo"]
            di = self._ir["DriverInfo"]

            driver_idx = di.get("DriverCarIdx", 0) if di else 0
            drivers = di.get("Drivers", []) if di else []

            car_name = ""
            car_id = 0
            if drivers and driver_idx < len(drivers):
                car_name = drivers[driver_idx].get("CarScreenName", "")
                car_id = drivers[driver_idx].get("CarID", 0)

            track_name = wi.get("TrackDisplayName", "") if wi else ""
            track_id = wi.get("TrackID", 0) if wi else 0
            track_len = wi.get("TrackLengthOfficial", "0 km") if wi else "0 km"

            # Parse track length string like "3.70 km"
            try:
                track_length_km = float(track_len.split()[0])
            except (ValueError, IndexError):
                track_length_km = 0.0

            pit_speed = wi.get("TrackPitSpeedLimit", "0 kph") if wi else "0 kph"
            try:
                pit_speed_kph = float(pit_speed.split()[0])
                pit_speed_mps = pit_speed_kph / 3.6
            except (ValueError, IndexError):
                pit_speed_mps = 0.0

            sessions = si.get("Sessions", []) if si else []
            session_type = ""
            if sessions:
                # Use the active session
                session_num = self._ir["SessionNum"] or 0
                if session_num < len(sessions):
                    session_type = sessions[session_num].get("SessionType", "")

            self._session_info = SessionInfo(
                car_name=car_name,
                car_id=car_id,
                track_name=track_name,
                track_id=track_id,
                track_length_km=track_length_km,
                session_type=session_type,
                pit_speed_limit_mps=pit_speed_mps,
                driver_idx=driver_idx,
            )
            self.session_info_updated.emit(self._session_info)
        except Exception as e:
            print(f"[telemetry] session info parse error: {e}")

    def _parse_competitors(self):
        """Parse competitor positions and pit status."""
        if self._ir is None or not self._connected:
            return

        try:
            di = self._ir["DriverInfo"]
            if not di:
                return

            drivers = di.get("Drivers", [])
            driver_idx = di.get("DriverCarIdx", 0)
            competitors = []

            for i, d in enumerate(drivers):
                if i == driver_idx:
                    continue
                if d.get("CarIsPaceCar", 0):
                    continue

                comp = CompetitorInfo(
                    car_idx=i,
                    position=d.get("CarClassPosition", 0),
                    class_position=d.get("CarClassPosition", 0),
                    laps_completed=self._ir["CarIdxLap"] or 0
                    if isinstance(self._ir["CarIdxLap"], int)
                    else (self._ir["CarIdxLap"][i] if self._ir["CarIdxLap"] else 0),
                    on_pit_road=bool(
                        self._ir["CarIdxOnPitRoad"][i]
                        if self._ir["CarIdxOnPitRoad"]
                        else False
                    ),
                )
                competitors.append(comp)

            self.competitors_updated.emit(competitors)
        except Exception as e:
            print(f"[telemetry] competitor parse error: {e}")
