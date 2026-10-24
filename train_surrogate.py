"""Train the neural surrogate of the plant dynamics (next-step prediction).

Trained on NORMAL operation only: the RL environment explores small deviations around the
nominal steady state, which is where this model is valid. It does not model faults.

Known limitation: in the dataset the XMVs are outputs of the plant's PID controllers, so the
model learns correlations between valves and measurements, not pure causal valve responses.
The script reports the persistence baseline (predict x[t+1] = x[t]) so the model can be
judged against something meaningful.
"""
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Input, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam

from common import WindowSequence, log, save_json, set_seed, setup_logging
from config import MODELS_DIR, N_FEATURES, SURROGATE_PATH
from data_processor import load_split, run_split_masks

BATCH_SIZE = 128
EPOCHS = 30


def build_surrogate():
    model = Sequential([
        Input(shape=(None, N_FEATURES)),  # flexible sequence length
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=True),
        TimeDistributed(Dense(N_FEATURES)),
    ])
    model.compile(optimizer=Adam(1e-3, clipnorm=1.0), loss="mse")
    return model


def train():
    setup_logging()
    set_seed()
    MODELS_DIR.mkdir(exist_ok=True)
    X, y, runs, _ = load_split("train")
    tr_mask, val_mask = run_split_masks(runs)
    normal = y == 0
    tr_idx, val_idx = np.where(normal & tr_mask)[0], np.where(normal & val_mask)[0]
    log.info("normal windows: train=%d val=%d", len(tr_idx), len(val_idx))

    model = build_surrogate()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(SURROGATE_PATH, save_best_only=True, monitor="val_loss"),
    ]
    model.fit(WindowSequence(X, tr_idx, batch_size=BATCH_SIZE, mode="surrogate", workers=1),
              validation_data=WindowSequence(X, val_idx, batch_size=BATCH_SIZE, mode="surrogate", shuffle=False),
              epochs=EPOCHS, callbacks=callbacks)

    xv = np.asarray(X[np.sort(val_idx)], dtype=np.float32)
    model_mse = float(np.mean(np.square(model.predict(xv[:, :-1], verbose=0) - xv[:, 1:])))
    persistence_mse = float(np.mean(np.square(xv[:, :-1] - xv[:, 1:])))
    log.info("held-out MSE: surrogate=%.4f persistence=%.4f", model_mse, persistence_mse)
    save_json(MODELS_DIR / "surrogate_metrics.json",
              {"surrogate_mse": model_mse, "persistence_mse": persistence_mse})


if __name__ == "__main__":
    train()
