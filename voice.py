"""Voice spotter for LapEdge — spoken callouts via pyttsx3."""

import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from queue import PriorityQueue, Empty

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class VoicePriority(IntEnum):
    CRITICAL = 0
    ADVISORY = 1
    INFO = 2
    STATUS = 3


@dataclass(order=True)
class SpeechRequest:
    priority: int              # VoicePriority int; compared first (lower = higher priority)
    enqueue_time: float        # tie-break: older request wins
    text: str = field(compare=False)
    dedupe_key: str = field(default="", compare=False)


class VoiceWorker(QObject):
    """Processes a priority queue of TTS requests in a dedicated thread."""

    speech_started = pyqtSignal(str)
    voice_error = pyqtSignal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config.voice
        self._queue: PriorityQueue = PriorityQueue()
        self._engine = None
        self._running = False
        self._recent: dict = {}  # dedupe_key -> last spoken timestamp
        self._interrupt = threading.Event()

    @pyqtSlot()
    def start(self):
        """Initialise pyttsx3. Must be called from within the voice thread."""
        if pyttsx3 is None:
            print("[voice] pyttsx3 not available — voice disabled")
            self._running = False
            return
        if not self._config.enabled:
            print("[voice] voice disabled in config")
            self._running = False
            return
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("volume", self._config.volume)
            self._engine.setProperty("rate", self._config.rate)
            if self._config.voice_id:
                self._engine.setProperty("voice", self._config.voice_id)
            self._running = True
            print("[voice] VoiceWorker started")
        except Exception as e:
            print(f"[voice] pyttsx3 init failed: {e}")
            self._running = False
            self.voice_error.emit(str(e))

    def stop(self):
        """Signal the worker to stop."""
        self._running = False
        self.stop_playback()

    @pyqtSlot()
    def stop_playback(self):
        """Interrupt any ongoing utterance. Runs on the voice thread via queued connection."""
        self._interrupt.set()
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception as e:
                print(f"[voice] error stopping engine: {type(e).__name__}: {e}")

    @pyqtSlot(str, int)
    def request_speech(self, text: str, priority: int):
        """Enqueue a speech request. Called from main thread via queued connection."""
        if not self._running:
            return
        if not self._config.enabled:
            return
        p = VoicePriority(priority)
        # CRITICAL items are never deduplicated; all others dedupe by text
        dedupe_key = "" if p == VoicePriority.CRITICAL else text
        req = SpeechRequest(
            priority=int(p),
            enqueue_time=time.time(),
            text=text,
            dedupe_key=dedupe_key,
        )
        self._queue.put(req)

    @pyqtSlot()
    def process_queue(self):
        """Drain the priority queue and speak the highest-priority item."""
        if not self._running or self._engine is None:
            return

        now = time.time()
        candidates = []

        # Drain entire queue
        while True:
            try:
                req = self._queue.get_nowait()
                candidates.append(req)
            except Empty:
                break

        best = None
        valid_candidates = []
        for req in candidates:
            # Discard stale non-CRITICAL items (queued > 10s ago)
            if req.priority != VoicePriority.CRITICAL:
                if now - req.enqueue_time > 10.0:
                    continue

            # Deduplication check (CRITICAL items have dedupe_key="", skip check)
            if req.dedupe_key:
                last_spoken = self._recent.get(req.dedupe_key, 0.0)
                if now - last_spoken < self._config.min_repeat_interval_s:
                    continue

            valid_candidates.append(req)

            # Pick highest priority (lowest int value)
            if best is None or req.priority < best.priority:
                best = req

        # Re-enqueue valid candidates that were not selected
        for req in valid_candidates:
            if req is not best:
                self._queue.put_nowait(req)

        if best is not None:
            if best.priority == VoicePriority.CRITICAL:
                self.stop_playback()
            spoken = self._speak_blocking(best.text)
            if best.dedupe_key and spoken:
                self._recent[best.dedupe_key] = now

    def _speak_blocking(self, text: str) -> bool:
        """Blocking TTS call — intentionally blocks the voice thread only.

        Returns True if speech completed without error, False otherwise.
        """
        self._interrupt.clear()
        try:
            self.speech_started.emit(text)
            self._engine.say(text)
            self._engine.runAndWait()
            return True
        except RuntimeError as e:
            print(f"[voice] speech error: RuntimeError: {e}")
            self.voice_error.emit(f"RuntimeError: {e}")
            return False
        except Exception as e:
            print(f"[voice] speech error: {type(e).__name__}: {e}")
            self.voice_error.emit(f"{type(e).__name__}: {e}")
            return False
