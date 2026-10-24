"""Train the CNN-LSTM fault classifier (21 classes). Validation is split by simulationRun."""
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Conv1D, Dense, Dropout, Input, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam

from common import WindowSequence, class_weights, log, save_json, set_seed, setup_logging
from config import CLASSIFIER_PATH, MODELS_DIR, N_FEATURES, NUM_CLASSES, WINDOW_SIZE
from data_processor import load_split, run_split_masks

BATCH_SIZE = 64
EPOCHS = 30


def build_classifier(window_size=WINDOW_SIZE, features=N_FEATURES, num_classes=NUM_CLASSES):
    model = Sequential([
        Input(shape=(window_size, features)),
        Conv1D(64, 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train():
    setup_logging()
    set_seed()
    MODELS_DIR.mkdir(exist_ok=True)
    X, y, runs, _ = load_split("train")
    tr_mask, val_mask = run_split_masks(runs)
    tr_idx, val_idx = np.where(tr_mask)[0], np.where(val_mask)[0]
    weights = class_weights(y[tr_idx], NUM_CLASSES)
    log.info("windows: train=%d val=%d", len(tr_idx), len(val_idx))

    model = build_classifier()
    model.summary(print_fn=log.info)
    train_seq = WindowSequence(X, tr_idx, y, BATCH_SIZE, "classifier", True, NUM_CLASSES, weights, workers=1)
    val_seq = WindowSequence(X, val_idx, y, BATCH_SIZE, "classifier", False, NUM_CLASSES)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(CLASSIFIER_PATH, save_best_only=True, monitor="val_loss"),
    ]
    hist = model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS, callbacks=callbacks)
    save_json(MODELS_DIR / "classifier_history.json", hist.history)


if __name__ == "__main__":
    train()
