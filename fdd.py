"""Fault detection & diagnosis building blocks (no I/O, no UI)."""
import os
from collections import deque
from dataclasses import dataclass, field

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from config import AUTOENCODER_PATH, CLASSIFIER_PATH, FEATURE_NAMES, RL_BEST_PATH, SCALER_PATH, THRESHOLD_PATH  # noqa: E402


@dataclass
class Models:
    """Everything that is expensive to load. Share one instance across sessions."""
    scaler: object
    autoencoder: keras.Model
    classifier: keras.Model
    threshold: float
    rl_agent: object = None
    _ae_fn: object = field(default=None, repr=False)
    _clf_fn: object = field(default=None, repr=False)

    def __post_init__(self):
        self._ae_fn = tf.function(lambda x: self.autoencoder(x, training=False))
        self._clf_fn = tf.function(lambda x: self.classifier(x, training=False))

    @classmethod
    def load(cls, load_rl=True):
        rl = None
        if load_rl and RL_BEST_PATH.exists():
            from stable_baselines3 import PPO
            rl = PPO.load(RL_BEST_PATH)
        return cls(
            scaler=joblib.load(SCALER_PATH),
            autoencoder=keras.models.load_model(AUTOENCODER_PATH),
            classifier=keras.models.load_model(CLASSIFIER_PATH),
            threshold=float(joblib.load(THRESHOLD_PATH)),
            rl_agent=rl,
        )

    def reconstruction_error(self, window: np.ndarray):
        """window: (T, F) scaled. Returns (mse, per_feature_mse)."""
        x = window[None].astype(np.float32)
        rec = self._ae_fn(tf.constant(x)).numpy()
        sq = np.square(x - rec)[0]
        return float(sq.mean()), sq.mean(axis=0)

    def class_probs(self, window: np.ndarray) -> np.ndarray:
        return self._clf_fn(tf.constant(window[None].astype(np.float32))).numpy()[0]


class AlarmFilter:
    """Debounce: raise after `confirm` consecutive exceedances, clear after `clear` consecutive normals."""

    def __init__(self, confirm: int = 3, clear: int = 10):
        self.confirm, self.clear = confirm, clear
        self.active = False
        self._hits = 0
        self._misses = 0

    def update(self, exceeded: bool) -> bool:
        if exceeded:
            self._hits, self._misses = self._hits + 1, 0
            if self._hits >= self.confirm:
                self.active = True
        else:
            self._misses, self._hits = self._misses + 1, 0
            if self._misses >= self.clear:
                self.active = False
        return self.active

    def reset(self):
        self.active, self._hits, self._misses = False, 0, 0


class DiagnosisSmoother:
    """Averages the classifier's probabilities over the last `n` windows of an alarm episode."""

    def __init__(self, n: int = 10):
        self._probs = deque(maxlen=n)

    def update(self, probs: np.ndarray) -> np.ndarray:
        self._probs.append(probs)
        return np.mean(self._probs, axis=0)

    def reset(self):
        self._probs.clear()


def top_deviations(window: np.ndarray, scaler, k: int = 5, recent: int = 10):
    """Variables whose recent mean deviates most from nominal (scaled units = standard deviations)."""
    z = window[-recent:].mean(axis=0)
    order = np.argsort(-np.abs(z))[:k]
    real = scaler.inverse_transform(window[-recent:].mean(axis=0, keepdims=True))[0]
    return [
        {"variable": FEATURE_NAMES[i], "index": int(i), "z": float(z[i]), "value": float(real[i]),
         "nominal": float(scaler.mean_[i])}
        for i in order
    ]


def format_evidence(devs, mse: float, threshold: float) -> str:
    lines = [f"Reconstruction error {mse:.3f} (alarm threshold {threshold:.3f}). "
             "Largest deviations from nominal when the alarm was raised:"]
    for d in devs:
        lines.append(f"- {d['variable']}: {d['value']:.4g} (nominal {d['nominal']:.4g}, {d['z']:+.1f} sigma)")
    return "\n".join(lines)
