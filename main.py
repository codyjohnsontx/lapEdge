"""LapEdge — Real-time iRacing pit strategy advisor overlay.

Entry point: wires telemetry capture, data processing, ML inference,
strategy engine, and overlay UI into a single pipeline.
"""

import signal
import sys
import time

from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtWidgets import QApplication

from config import load_config, save_config, AppConfig
from telemetry import TelemetryWorker, TelemetryFrame, SessionInfo, SessionFlags
from data_processor import DataProcessor
from strategy_engine import StrategyEngine
from model import ModelWorker
from overlay import OverlayWidget
from logger import TelemetryLogger
from voice import VoiceWorker, VoicePriority


class LapEdgeApp(QObject):
    """Main application controller. Owns all modules and manages lifecycle."""

    _speak = pyqtSignal(str, int)  # cross-thread speech requests

    def __init__(self, config: AppConfig):
        super().__init__()
        self._config = config
        self._last_ui_update = 0.0
        self._ui_interval = 1.0 / config.telemetry.ui_update_rate_hz

        # --- Modules ---
        self._data_processor = DataProcessor(config)
        self._strategy_engine = StrategyEngine()

        # Logger (runs on main thread, buffered writes)
        self._logger = TelemetryLogger(config)

        # Overlay
        self._overlay = OverlayWidget(config)

        # Telemetry worker + thread
        self._telemetry_thread = QThread()
        self._telemetry_worker = TelemetryWorker(config)
        self._telemetry_worker.moveToThread(self._telemetry_thread)

        # Model worker + thread
        self._model_thread = QThread()
        self._model_worker = ModelWorker(config)
        self._model_worker.moveToThread(self._model_thread)

        # Model queue processing timer (runs in model thread)
        self._model_timer = QTimer()
        self._model_timer.setInterval(100)  # 10hz check
        self._model_timer.moveToThread(self._model_thread)
        self._model_timer.timeout.connect(self._model_worker.process_queue)

        # Voice worker + thread
        self._voice_thread = QThread()
        self._voice_worker = VoiceWorker(config)
        self._voice_worker.moveToThread(self._voice_thread)

        # Voice queue processing timer (runs in voice thread)
        self._voice_timer = QTimer()
        self._voice_timer.setInterval(200)
        self._voice_timer.moveToThread(self._voice_thread)
        self._voice_timer.timeout.connect(self._voice_worker.process_queue)

        # Voice state tracking
        self._prev_on_pit_road: bool = False
        self._prev_session_flags: int = 0
        self._session_greeted: bool = False

        # Hotkey thread
        self._hotkey_listener = None

        self._connect_signals()

    def _connect_signals(self):
        """Wire the entire signal/slot pipeline."""

        # Telemetry thread lifecycle
        self._telemetry_thread.started.connect(self._telemetry_worker.start)

        # Model thread lifecycle
        self._model_thread.started.connect(self._model_worker.start)
        self._model_thread.started.connect(self._model_timer.start)

        # Voice thread lifecycle
        self._voice_thread.started.connect(self._voice_worker.start)
        self._voice_thread.started.connect(self._voice_timer.start)
        self._speak.connect(self._voice_worker.request_speech)
        self._telemetry_worker.connection_status.connect(self._on_connection_voice)

        # Telemetry → Logger
        self._telemetry_worker.frame_ready.connect(self._logger.log_frame)

        # Telemetry → Main thread handlers
        self._telemetry_worker.frame_ready.connect(self._on_frame)
        self._telemetry_worker.lap_completed.connect(self._on_lap_completed)
        self._telemetry_worker.session_info_updated.connect(self._on_session_info)
        self._telemetry_worker.competitors_updated.connect(
            self._data_processor.set_competitors
        )
        self._telemetry_worker.connection_status.connect(
            self._overlay.update_connection
        )

        # Model → Strategy → Overlay
        self._model_worker.prediction_ready.connect(self._on_prediction)

    def start(self):
        """Start all threads and show overlay."""
        print("[main] Starting LapEdge...")

        self._overlay.show()

        # Start threads
        self._telemetry_thread.start()
        self._model_thread.start()
        self._voice_thread.start()

        # Start hotkey listener
        self._start_hotkeys()

        print("[main] LapEdge running. Ctrl+C to quit.")

    def _start_hotkeys(self):
        """Start global hotkey listener in a daemon thread."""
        try:
            from pynput.keyboard import GlobalHotKeys

            hotkeys = GlobalHotKeys({
                self._config.hotkeys.toggle_overlay: self._hotkey_toggle,
                self._config.hotkeys.cycle_mode: self._hotkey_cycle_mode,
                self._config.hotkeys.request_update: self._hotkey_request_update,
            })
            hotkeys.daemon = True
            hotkeys.start()
            self._hotkey_listener = hotkeys
            print("[main] Global hotkeys registered")
        except Exception as e:
            print(f"[main] Hotkey setup failed (non-fatal): {e}")

    def _hotkey_toggle(self):
        """Toggle overlay visibility (called from hotkey thread)."""
        QTimer.singleShot(0, self._toggle_overlay)

    def _hotkey_cycle_mode(self):
        """Cycle alert mode (called from hotkey thread)."""
        QTimer.singleShot(0, self._overlay._cycle_mode)

    def _hotkey_request_update(self):
        """Force show current recommendation (called from hotkey thread)."""
        QTimer.singleShot(0, self._overlay.force_show_recommendation)

    def _toggle_overlay(self):
        if self._overlay.isVisible():
            self._overlay.hide()
        else:
            self._overlay.show()

    # --- Signal handlers ---

    @pyqtSlot(object)
    def _on_frame(self, frame: TelemetryFrame):
        """Handle each telemetry frame. Throttle UI updates to 4hz."""
        # State-change checks run on every frame (not throttled)
        self._check_pit_road(frame)
        self._check_flags(frame)

        now = time.time()
        if now - self._last_ui_update < self._ui_interval:
            return
        self._last_ui_update = now

        # Buffer for per-lap aggregation
        self._data_processor.buffer_frame(frame)

        # Update live telemetry display
        self._overlay.update_live_telemetry(
            frame.speed, frame.fuel_pct, frame.gear, frame.rpm
        )

    @pyqtSlot(object, float)
    def _on_lap_completed(self, frame: TelemetryFrame, lap_time: float):
        """Handle lap completion — run the full strategy pipeline."""
        if lap_time <= 0:
            return

        summary, context, features = self._data_processor.process_lap(
            frame, lap_time
        )

        # Update stats display
        gap_text = f"{context.gap_ahead_s:.1f}s / {context.gap_behind_s:.1f}s"
        self._overlay.update_stats(
            context.current_lap,
            context.fuel_laps_remaining,
            context.estimated_tire_life_pct,
            gap_text,
        )

        # ML path: queue prediction if model loaded
        if self._model_worker.model_available:
            self._model_worker.request_prediction(features.values)
        else:
            # Rule-based only
            rec = self._strategy_engine.evaluate(context, None)
            if self._overlay.should_show_recommendation(
                context.current_lap, rec
            ):
                self._overlay.update_recommendation(rec)
            self._speak_recommendation(rec, context)

    @pyqtSlot(object)
    def _on_session_info(self, info: SessionInfo):
        """Handle session info updates."""
        self._data_processor.set_session_info(info)
        self._logger.start_session(info.car_name, info.track_name)

        # Load car-specific model
        self._model_worker.load_for_car(info.car_name)

        # Session greeting (once per connection)
        if not self._session_greeted and info.track_name:
            self._session_greeted = True
            self._speak.emit(
                f"LapEdge active. Racing at {info.track_name}.",
                int(VoicePriority.STATUS),
            )

        print(f"[main] Session: {info.car_name} @ {info.track_name} "
              f"({info.session_type})")

    @pyqtSlot(object)
    def _on_prediction(self, prediction):
        """Handle ML model prediction — evaluate with strategy engine."""
        # Need current race context — use the last processed context
        if self._data_processor._lap_summaries:
            last_summary = self._data_processor._lap_summaries[-1]
            # Rebuild context from last known state
            # (simplified — in production, cache the last context)
            rec = self._strategy_engine.evaluate(
                self._data_processor._build_race_context(
                    TelemetryFrame(lap=last_summary.lap_number),
                    last_summary,
                ),
                prediction,
            )
            self._overlay.update_recommendation(rec)
            self._speak_recommendation(rec, None)

    @pyqtSlot(bool)
    def _on_connection_voice(self, connected: bool):
        """Speak connection status changes."""
        if connected:
            self._session_greeted = False  # allow greeting on next session info
            self._speak.emit("Connected to iRacing.", int(VoicePriority.STATUS))
        else:
            self._speak.emit("iRacing disconnected.", int(VoicePriority.STATUS))

    def _check_pit_road(self, frame: TelemetryFrame):
        """Speak when the car enters or exits pit road."""
        if frame.on_pit_road == self._prev_on_pit_road:
            return
        self._prev_on_pit_road = frame.on_pit_road
        msg = "Entering pit road." if frame.on_pit_road else "Exiting pit road."
        self._speak.emit(msg, int(VoicePriority.ADVISORY))

    def _check_flags(self, frame: TelemetryFrame):
        """Speak flag changes."""
        new, old = frame.session_flags, self._prev_session_flags
        if new == old:
            return
        self._prev_session_flags = new
        newly_set = new & ~old
        if newly_set & SessionFlags.GREEN:
            self._speak.emit("Green flag.", int(VoicePriority.ADVISORY))
        elif newly_set & (SessionFlags.YELLOW | SessionFlags.CAUTION):
            self._speak.emit("Yellow flag. Caution.", int(VoicePriority.CRITICAL))
        elif newly_set & SessionFlags.RED:
            self._speak.emit("Red flag. Session stopped.", int(VoicePriority.CRITICAL))
        elif newly_set & SessionFlags.CHECKERED:
            self._speak.emit("Checkered flag. Session complete.", int(VoicePriority.ADVISORY))

    def _speak_recommendation(self, rec, context):
        """Speak a strategy recommendation based on urgency and voice config."""
        from strategy_engine import Urgency
        if not self._config.voice.enabled:
            return
        if rec.urgency == Urgency.CRITICAL:
            self._speak.emit(rec.message, int(VoicePriority.CRITICAL))
        elif rec.urgency == Urgency.ADVISORY and self._config.voice.speak_advisory:
            self._speak.emit(rec.message, int(VoicePriority.ADVISORY))
        elif rec.urgency == Urgency.INFO and self._config.voice.speak_info:
            self._speak.emit(rec.message, int(VoicePriority.INFO))

    def shutdown(self):
        """Graceful shutdown: stop workers, wait for threads, save config."""
        print("[main] Shutting down...")

        # Save overlay position
        self._overlay.save_position(self._config)
        save_config(self._config)

        # Stop workers
        self._telemetry_worker.stop()
        self._model_worker.stop()
        self._model_timer.stop()
        self._voice_worker.stop()
        self._voice_timer.stop()
        self._logger.stop()

        # Stop hotkey listener
        if self._hotkey_listener:
            self._hotkey_listener.stop()

        # Wait for threads
        self._telemetry_thread.quit()
        self._telemetry_thread.wait(3000)
        self._model_thread.quit()
        self._model_thread.wait(3000)
        self._voice_thread.quit()
        self._voice_thread.wait(3000)

        print("[main] LapEdge stopped.")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LapEdge")

    config = load_config()
    lap_edge = LapEdgeApp(config)

    # Ctrl+C handling
    def sigint_handler(*args):
        lap_edge.shutdown()
        app.quit()

    signal.signal(signal.SIGINT, sigint_handler)

    # Timer to allow Python signal handling in Qt event loop
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(500)

    lap_edge.start()

    exit_code = app.exec_()
    lap_edge.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
