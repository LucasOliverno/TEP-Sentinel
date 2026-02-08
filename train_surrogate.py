import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import gc

# Settings
FEATURES = 52 # 41 States + 11 Actions
BATCH_SIZE = 128
EPOCHS = 10
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_surrogate_model():
    """
    LSTM Predictor:
    Input: Sequence of States [t_0 ... t_n]
    Output: Sequence of Next States [t_1 ... t_{n+1}]
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(None, FEATURES)), # None allows flexible window size
        LSTM(64, return_sequences=True),
        TimeDistributed(Dense(FEATURES)) # Predict all 52 vars for next step
    ])
    
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mse')
    return model

def train_surrogate():
    print("Loading tep_train.npz...")
    try:
        data = np.load('processed_data/tep_train.npz')
        X_all = data['X'] # (N, 100, 52)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Preparing Surrogate Data (Next-Step Prediction)...")
    # Only use a subset to save memory if needed, but TEP isn't too huge for this (800MB)
    # Strategy: 
    # Input: Steps 0 to 98
    # Target: Steps 1 to 99
    X_train = X_all[:, :-1, :] # (N, 99, 52)
    y_train = X_all[:, 1:, :]  # (N, 99, 52)
    
    # We only train on NORMAL data to learn the "Physics" of the plant.
    # Training on Faulty data might make the simulator think faults are "natural dynamics" rather than external events.
    # Wait, actually we WANT the simulator to know how the plant reacts under faults if we pass the fault code?
    # No, the RL agent controls the VALVES (XMV). The 'Fault' is an external disturbance.
    # For robust control, we just train on Normal dynamics mostly.
    # Actually, including faulty data allows the surrogate to learn how the plant spirals out of control, which is useful!
    # But for now, let's stick to ALL data to capture all dynamics.
    
    print(f"Training Shape: {X_train.shape}")
    
    model = build_surrogate_model()
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(f"{MODEL_DIR}/tep_surrogate.keras", save_best_only=True)
    ]
    
    print("Starting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        shuffle=True
    )
    
    print("Surrogate Model Saved.")

if __name__ == "__main__":
    train_surrogate()
