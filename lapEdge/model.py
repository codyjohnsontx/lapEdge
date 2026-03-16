"""TensorFlow model loading and inference for LapEdge pit strategy."""

import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from data_processor import NUM_FEATURES


@dataclass
class ModelPrediction:
    """Output of the pit strategy ML model."""
    pit_lap_offset: int = 0       # laps until recommended pit (0 = now)
    pit_window: tuple = (0, 0)    # (earliest, latest) pit lap range
    compound: str = "hard"        # recommended tire compound
    confidence: float = 0.0       # 0-1 model confidence
    model_type: str = "none"      # "tf", "rule_fallback", "none"


class PitStrategyModel:
    """Wraps a TensorFlow model for pit strategy inference.

    Architecture:
        Input(24) → Dense(64,relu) → BN → Dropout(0.3)
                   → Dense(32,relu) → BN → Dropout(0.2)
                   → Dense(16,relu)
                   → 3 heads:
                       pit_lap_offset(1, linear)
                       compound_probs(3, softmax)
                       confidence(1, sigmoid)

    <10K parameters, <10ms inference on CPU.
    """

    COMPOUNDS = ["soft", "medium", "hard"]

    def __init__(self, model_dir: str = "models"):
        self._model_dir = Path(model_dir)
        self._model = None
        self._loaded = False
        self._tf = None

    def load(self, car_name: str = "") -> bool:
        """Load model weights for a specific car. Returns False if unavailable."""
        try:
            import tensorflow as tf
            self._tf = tf
        except ImportError:
            print("[model] TensorFlow not available — using rule-based fallback")
            return False

        # Look for car-specific model first, then generic
        safe_name = car_name.lower().replace(" ", "_")[:30]
        candidates = [
            self._model_dir / f"{safe_name}.keras",
            self._model_dir / f"{safe_name}.h5",
            self._model_dir / "pit_strategy.keras",
            self._model_dir / "pit_strategy.h5",
        ]

        for path in candidates:
            if path.exists():
                try:
                    self._model = tf.keras.models.load_model(str(path))
                    self._loaded = True
                    print(f"[model] Loaded model from {path}")
                    return True
                except Exception as e:
                    print(f"[model] Failed to load {path}: {e}")

        print("[model] No model weights found — using rule-based fallback")
        return False

    def predict(self, features: np.ndarray) -> Optional[ModelPrediction]:
        """Run inference on a 24-element feature vector."""
        if not self._loaded or self._model is None:
            return None

        if features.shape != (NUM_FEATURES,):
            print(f"[model] Expected shape ({NUM_FEATURES},), got {features.shape}")
            return None

        try:
            # Model expects batch dimension
            x = features.reshape(1, NUM_FEATURES)
            outputs = self._model.predict(x, verbose=0)

            # Parse multi-head output
            if isinstance(outputs, list) and len(outputs) == 3:
                pit_offset_raw = float(outputs[0][0, 0])
                compound_probs = outputs[1][0]
                confidence = float(outputs[2][0, 0])
            else:
                # Single output fallback: first element is offset, rest compound
                out = outputs[0] if isinstance(outputs, list) else outputs
                pit_offset_raw = float(out[0, 0])
                compound_probs = out[0, 1:4] if out.shape[1] >= 4 else np.array([0, 0, 1])
                confidence = float(out[0, -1]) if out.shape[1] >= 5 else 0.5

            pit_lap_offset = max(0, round(pit_offset_raw))
            compound_idx = int(np.argmax(compound_probs))
            compound = self.COMPOUNDS[compound_idx] if compound_idx < len(self.COMPOUNDS) else "hard"
            confidence = max(0.0, min(1.0, confidence))

            return ModelPrediction(
                pit_lap_offset=pit_lap_offset,
                pit_window=(max(0, pit_lap_offset - 2), pit_lap_offset + 2),
                compound=compound,
                confidence=confidence,
                model_type="tf",
            )
        except Exception as e:
            print(f"[model] Inference error: {e}")
            return None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def build_model():
        """Build the model architecture (for training, not used at runtime).

        Returns a compiled tf.keras.Model.
        """
        import tensorflow as tf

        inputs = tf.keras.Input(shape=(NUM_FEATURES,), name="features")
        x = tf.keras.layers.Dense(64, activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)

        pit_offset = tf.keras.layers.Dense(1, activation="linear", name="pit_offset")(x)
        compound = tf.keras.layers.Dense(3, activation="softmax", name="compound")(x)
        confidence = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence")(x)

        model = tf.keras.Model(inputs=inputs, outputs=[pit_offset, compound, confidence])
        model.compile(
            optimizer="adam",
            loss={
                "pit_offset": "mse",
                "compound": "categorical_crossentropy",
                "confidence": "binary_crossentropy",
            },
            loss_weights={"pit_offset": 1.0, "compound": 0.5, "confidence": 0.3},
        )
        return model


class ModelWorker(QObject):
    """Runs ML inference in a dedicated QThread, driven by a queue."""

    prediction_ready = pyqtSignal(object)  # ModelPrediction

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config.model
        self._model = PitStrategyModel(config.model.model_dir)
        self._queue = queue.Queue()
        self._running = False

    def start(self):
        """Initialize the model (lazy TF import happens here)."""
        self._running = True
        if self._config.enabled:
            # Model loading happens on first car identification
            pass

    def load_for_car(self, car_name: str):
        """Load model weights for a specific car."""
        if self._config.enabled:
            self._model.load(car_name)

    def request_prediction(self, features: np.ndarray):
        """Queue a prediction request (called from main thread)."""
        if not self._running or not self._model.is_loaded:
            return
        self._queue.put(features)

    def process_queue(self):
        """Process pending prediction requests. Called by timer in model thread."""
        while not self._queue.empty():
            try:
                features = self._queue.get_nowait()
                prediction = self._model.predict(features)
                if prediction is not None:
                    self.prediction_ready.emit(prediction)
            except queue.Empty:
                break

    def stop(self):
        self._running = False

    @property
    def model_available(self) -> bool:
        return self._model.is_loaded
