"""PyQt5 transparent overlay UI for LapEdge."""

from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QPainter, QBrush, QPen
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizeGrip,
    QPushButton, QApplication,
)

from config import AppConfig, PASSIVE, ON_DEMAND, PERIODIC, ALERT_MODES
from strategy_engine import Recommendation, Urgency


# Color palette
COLORS = {
    Urgency.INFO: "#4CAF50",       # Green
    Urgency.ADVISORY: "#FF9800",   # Orange
    Urgency.CRITICAL: "#F44336",   # Red
}
BG_COLOR = QColor(20, 20, 30, 200)
TEXT_COLOR = "#E0E0E0"
HEADER_COLOR = "#FFFFFF"


class OverlayWidget(QWidget):
    """Frameless, always-on-top, translucent pit strategy overlay."""

    mode_changed = pyqtSignal(str)
    visibility_toggled = pyqtSignal(bool)

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self._config = config.overlay
        self._drag_pos = QPoint()
        self._connected = False
        self._current_mode = config.overlay.alert_mode
        self._recommendation: Recommendation = Recommendation(
            message="Waiting for data...", urgency=Urgency.INFO
        )
        self._last_lap_shown = 0

        self._setup_window()
        self._build_ui()

    def _setup_window(self):
        """Configure window flags for overlay behavior."""
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(self._config.opacity)
        self.setGeometry(
            self._config.x, self._config.y,
            self._config.width, self._config.height
        )
        self.setMinimumSize(300, 200)

    def _build_ui(self):
        """Build the overlay layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Header row: title + connection indicator
        header = QHBoxLayout()
        self._title = QLabel("LapEdge")
        self._title.setStyleSheet(
            f"color: {HEADER_COLOR}; font-weight: bold; font-size: 16px;"
        )
        header.addWidget(self._title)
        header.addStretch()

        self._conn_indicator = QLabel("\u25CF")  # Filled circle
        self._conn_indicator.setStyleSheet("color: #F44336; font-size: 14px;")
        header.addWidget(self._conn_indicator)

        self._mode_label = QLabel(self._current_mode.upper())
        self._mode_label.setStyleSheet(
            f"color: {TEXT_COLOR}; font-size: 10px; padding-left: 4px;"
        )
        header.addWidget(self._mode_label)
        layout.addLayout(header)

        # Recommendation display
        self._rec_label = QLabel("Waiting for data...")
        self._rec_label.setWordWrap(True)
        self._rec_label.setStyleSheet(
            f"color: {COLORS[Urgency.INFO]}; font-size: 18px; font-weight: bold;"
        )
        layout.addWidget(self._rec_label)

        # Detail text
        self._detail_label = QLabel("")
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet(
            f"color: {TEXT_COLOR}; font-size: 12px;"
        )
        layout.addWidget(self._detail_label)

        # Stats row
        stats = QHBoxLayout()
        self._lap_label = QLabel("Lap: --")
        self._fuel_label = QLabel("Fuel: --")
        self._tire_label = QLabel("Tires: --")
        self._gap_label = QLabel("Gap: --")

        for label in [self._lap_label, self._fuel_label,
                      self._tire_label, self._gap_label]:
            label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11px;")
            stats.addWidget(label)

        layout.addLayout(stats)

        # Mode toggle button
        btn_row = QHBoxLayout()
        self._mode_btn = QPushButton("Mode")
        self._mode_btn.setFixedSize(60, 22)
        self._mode_btn.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,30); color: #CCC; "
            "border: 1px solid rgba(255,255,255,50); border-radius: 3px; "
            "font-size: 10px; }"
            "QPushButton:hover { background: rgba(255,255,255,60); }"
        )
        self._mode_btn.clicked.connect(self._cycle_mode)
        btn_row.addWidget(self._mode_btn)
        btn_row.addStretch()

        # Resize grip
        grip = QSizeGrip(self)
        grip.setStyleSheet("background: transparent;")
        btn_row.addWidget(grip, 0, Qt.AlignBottom | Qt.AlignRight)
        layout.addLayout(btn_row)

    def paintEvent(self, event):
        """Draw rounded semi-transparent background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(BG_COLOR))
        painter.setPen(QPen(QColor(60, 60, 80, 150), 1))
        painter.drawRoundedRect(self.rect(), 10, 10)
        painter.end()

    def mousePressEvent(self, event):
        """Enable dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Handle drag movement."""
        if event.buttons() == Qt.LeftButton and not self._drag_pos.isNull():
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    # --- Public slots ---

    @pyqtSlot(object)
    def update_recommendation(self, rec: Recommendation):
        """Update the recommendation display."""
        self._recommendation = rec
        color = COLORS.get(rec.urgency, COLORS[Urgency.INFO])
        self._rec_label.setText(rec.message)
        self._rec_label.setStyleSheet(
            f"color: {color}; font-size: 18px; font-weight: bold;"
        )
        self._detail_label.setText(rec.detail)

        # Flash on critical
        if rec.urgency == Urgency.CRITICAL:
            self._flash_critical()

    @pyqtSlot(int, float, float, str)
    def update_stats(self, lap: int, fuel_laps: float,
                     tire_life_pct: float, gap_text: str):
        """Update the stats display."""
        self._lap_label.setText(f"Lap: {lap}")
        self._fuel_label.setText(f"Fuel: {fuel_laps:.1f}L")
        self._tire_label.setText(f"Tires: {tire_life_pct:.0%}")
        self._gap_label.setText(f"Gap: {gap_text}")

    @pyqtSlot(float, float, int, float)
    def update_live_telemetry(self, speed_mps: float, fuel_pct: float,
                              gear: int, rpm: float):
        """Update live telemetry display (throttled to 4hz by caller)."""
        speed_kph = speed_mps * 3.6
        self._title.setText(
            f"LapEdge  {speed_kph:.0f}kph  G{gear}  {rpm:.0f}rpm"
        )

    @pyqtSlot(bool)
    def update_connection(self, connected: bool):
        """Update the connection status indicator."""
        self._connected = connected
        color = "#4CAF50" if connected else "#F44336"
        self._conn_indicator.setStyleSheet(f"color: {color}; font-size: 14px;")

    def should_show_recommendation(self, lap: int, rec: Recommendation) -> bool:
        """Check if recommendation should be displayed based on alert mode."""
        if rec.urgency == Urgency.CRITICAL:
            return True  # Always show critical

        if self._current_mode == PASSIVE:
            return False

        if self._current_mode == ON_DEMAND:
            return False  # Only shown when hotkey pressed

        if self._current_mode == PERIODIC:
            interval = self._config.periodic_interval_laps
            if lap - self._last_lap_shown >= interval:
                self._last_lap_shown = lap
                return True
            return False

        return True

    def force_show_recommendation(self):
        """Show current recommendation regardless of mode (hotkey trigger)."""
        self.update_recommendation(self._recommendation)

    def _cycle_mode(self):
        """Cycle through alert modes."""
        idx = ALERT_MODES.index(self._current_mode)
        self._current_mode = ALERT_MODES[(idx + 1) % len(ALERT_MODES)]
        self._mode_label.setText(self._current_mode.upper())
        self.mode_changed.emit(self._current_mode)

    def _flash_critical(self):
        """Brief flash effect for critical alerts."""
        original_opacity = self.windowOpacity()
        self.setWindowOpacity(1.0)
        QTimer.singleShot(200, lambda: self.setWindowOpacity(original_opacity))

    def save_position(self, config: AppConfig):
        """Save current position/size to config."""
        pos = self.pos()
        size = self.size()
        config.overlay.x = pos.x()
        config.overlay.y = pos.y()
        config.overlay.width = size.width()
        config.overlay.height = size.height()
