"""Train the LSTM autoencoder on NORMAL windows only and calibrate the alarm threshold.

The threshold is the 99th percentile of the reconstruction error on HELD-OUT normal runs
(split by simulationRun), not on the training windows themselves.
"""
import joblib
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam

from common import WindowSequence, log, set_seed, setup_logging
from config import AUTOENCODER_PATH, MODELS_DIR, N_FEATURES, THRESHOLD_PATH, WINDOW_SIZE
from data_processor import load_split, run_split_masks

LATENT_DIM = 16
BATCH_SIZE = 64
EPOCHS = 40
THRESHOLD_PERCENTILE = 99.0


def build_autoencoder(window_size=WINDOW_SIZE, features=N_FEATURES, latent_dim=LATENT_DIM):
    inputs = Input(shape=(window_size, features))
    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(latent_dim)(x)  # bottleneck
    x = RepeatVector(window_size)(x)
    x = LSTM(latent_dim, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)
    outputs = TimeDistributed(Dense(features))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0), loss="mse")
    return model


def window_mse(model, X, indices, batch_size=256):
    out = []
    for s in range(0, len(indices), batch_size):
        x = np.asarray(X[np.sort(indices[s:s + batch_size])], dtype=np.float32)
        out.append(np.mean(np.square(x - model.predict(x, verbose=0)), axis=(1, 2)))
    return np.concatenate(out)


def train():
    setup_logging()
    set_seed()
    MODELS_DIR.mkdir(exist_ok=True)
    X, y, runs, _ = load_split("train")
    tr_mask, val_mask = run_split_masks(runs)
    normal = y == 0
    tr_idx, val_idx = np.where(normal & tr_mask)[0], np.where(normal & val_mask)[0]
    log.info("normal windows: train=%d val=%d", len(tr_idx), len(val_idx))

    model = build_autoencoder()
    model.summary(print_fn=log.info)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(AUTOENCODER_PATH, save_best_only=True, monitor="val_loss"),
    ]
    model.fit(WindowSequence(X, tr_idx, batch_size=BATCH_SIZE, mode="autoencoder", workers=1),
              validation_data=WindowSequence(X, val_idx, batch_size=BATCH_SIZE, mode="autoencoder", shuffle=False),
              epochs=EPOCHS, callbacks=callbacks)

    val_mse = window_mse(model, X, val_idx)
    threshold = float(np.percentile(val_mse, THRESHOLD_PERCENTILE))
    log.info("Threshold (P%.0f of held-out normal runs) = %.4f", THRESHOLD_PERCENTILE, threshold)
    joblib.dump(threshold, THRESHOLD_PATH)


if __name__ == "__main__":
    train()
