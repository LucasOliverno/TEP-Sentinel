"""Shared helpers for the training scripts."""
import json
import logging
import os

import numpy as np

from config import RANDOM_SEED

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402

log = logging.getLogger("tep")


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def set_seed(seed: int = RANDOM_SEED):
    keras.utils.set_random_seed(seed)  # python, numpy and backend RNGs


def class_weights(y: np.ndarray, num_classes: int) -> dict:
    """Balanced class weights; classes absent from y get weight 1."""
    counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float64)
    w = np.ones(num_classes)
    present = counts > 0
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    return {i: float(v) for i, v in enumerate(w)}


class WindowSequence(keras.utils.PyDataset):
    """Serves batches from a (possibly memory-mapped) window array without loading it all.

    mode: 'autoencoder' -> (x, x); 'classifier' -> (x, onehot(y), sample_weight);
          'surrogate'   -> (x[:, :-1], x[:, 1:]).
    """

    def __init__(self, X, indices, y=None, batch_size=64, mode="autoencoder", shuffle=True,
                 num_classes=None, weights=None, seed=RANDOM_SEED, **kwargs):
        super().__init__(**kwargs)
        self.X, self.y = X, y
        self.indices = np.asarray(indices)
        self.batch_size, self.mode, self.shuffle = batch_size, mode, shuffle
        self.num_classes, self.weights = num_classes, weights
        self.rng = np.random.default_rng(seed)
        if shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, i):
        b = np.sort(self.indices[i * self.batch_size:(i + 1) * self.batch_size])  # sorted -> faster mmap reads
        x = np.asarray(self.X[b], dtype=np.float32)
        np.nan_to_num(x, copy=False)
        if self.mode == "autoencoder":
            return x, x
        if self.mode == "surrogate":
            return x[:, :-1], x[:, 1:]
        yb = self.y[b].astype(np.int64)
        onehot = np.eye(self.num_classes, dtype=np.float32)[yb]
        if self.weights is None:
            return x, onehot
        return x, onehot, np.array([self.weights[c] for c in yb], dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
