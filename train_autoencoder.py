import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
import os
import gc

# Settings
WINDOW_SIZE = 100
FEATURES = 52
LATENT_DIM = 16
BATCH_SIZE = 64
EPOCHS = 20
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_autoencoder(window_size, features, latent_dim):
    """
    LSTM Autoencoder Architecture
    """
    # Encoder
    inputs = Input(shape=(window_size, features))
    # Encoder: LSTM with tanh (stable)
    encoded = LSTM(64, activation='tanh', return_sequences=True)(inputs)
    encoded = LSTM(latent_dim, activation='tanh', return_sequences=False)(encoded) # Bottleneck
    
    # Decoder
    decoded = RepeatVector(window_size)(encoded)
    decoded = LSTM(latent_dim, activation='tanh', return_sequences=True)(decoded)
    decoded = LSTM(64, activation='tanh', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(features))(decoded)
    
    model = Model(inputs, decoded)
    
    # Optimizer with Gradient Clipping to prevent NaNs
    opt = Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(optimizer=opt, loss='mse')
    return model

def train_model():
    print("Loading processed data (tep_train.npz)...")
    try:
        data = np.load('processed_data/tep_train.npz')
        X = data['X']
        y = data['y']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Filter ONLY Normal Data (Class 0)
    print("Filtering Normal data (Class 0)...")
    # Boolean indexing creates a copy, reducing memory if we delete original
    idxs = (y == 0)
    X_normal = X[idxs]
    print(f"Normal Samples: {X_normal.shape}")
    
    # Free up memory (The full dataset is huge)
    del data, X, y
    gc.collect()
    
    # Sanitize Data
    print("Sanitizing data (NaN/Inf check)...")
    if np.isnan(X_normal).any() or np.isinf(X_normal).any():
        print("WARNING: Data contains NaNs or Infs. Replacing with 0.")
        X_normal = np.nan_to_num(X_normal, nan=0.0, posinf=0.0, neginf=0.0)
    
    model = build_autoencoder(WINDOW_SIZE, FEATURES, LATENT_DIM)
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True),
        ModelCheckpoint(f"{MODEL_DIR}/tep_autoencoder.keras", save_best_only=True, monitor='val_loss')
    ]
    
    print("Starting Training...")
    # Train
    history = model.fit(
        X_normal, X_normal, 
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        shuffle=True
    )
    
    print("Training Complete.")
    
    # Calculate Threshold
    print("Calculating Anomaly Threshold...")
    # Predict in batches to avoid OOM
    reconstructions = model.predict(X_normal, batch_size=BATCH_SIZE)
    mse = np.mean(np.power(X_normal - reconstructions, 2), axis=(1, 2))
    
    # Set threshold at 99th percentile
    threshold = np.percentile(mse, 99)
    print(f"Threshold (99%): {threshold}")
    
    # Save Threshold
    with open(f"{MODEL_DIR}/threshold.txt", "w") as f:
        f.write(str(threshold))
        
    joblib.dump(threshold, f"{MODEL_DIR}/tep_threshold.pkl")
    print(f"Threshold saved to {MODEL_DIR}/tep_threshold.pkl")

if __name__ == "__main__":
    train_model()
