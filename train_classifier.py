import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os
import gc

# Settings
WINDOW_SIZE = 100
FEATURES = 52
NUM_CLASSES = 21 # 0=Normal, 1-20=Faults
BATCH_SIZE = 64
EPOCHS = 20
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_classifier(window_size, features, num_classes):
    """
    CNN-LSTM Classifier Architecture
    """
    model = Sequential([
        Input(shape=(window_size, features)),
        
        # CNN Layers (Feature Extraction)
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM Layers (Temporal Dynamics)
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        
        # Dense Layers (Classification)
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    print("Loading processed data (tep_train.npz)...")
    try:
        data = np.load('processed_data/tep_train.npz')
        X_train = data['X']
        y_train = data['y']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Sanitize Data (NaN/Inf check) based on previous experience
    print("Sanitizing data...")
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("WARNING: Data contains NaNs or Infs. Replacing...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # One-Hot Encoding
    print(f"Encoding labels (Classes={NUM_CLASSES})...")
    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    
    # Model
    model = build_classifier(WINDOW_SIZE, FEATURES, NUM_CLASSES)
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True),
        ModelCheckpoint(f"{MODEL_DIR}/tep_classifier.keras", save_best_only=True, monitor='val_loss')
    ]
    
    print("Starting Training...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2, # Note: Simple split. For time series, usually careful, but here samples are shuffled/windowed
        callbacks=callbacks,
        shuffle=True
    )
    
    print("Training Complete.")
    
    # Clean up memory
    del X_train, y_train, y_train_cat
    gc.collect()

if __name__ == "__main__":
    train_model()
