import numpy as np
import joblib
import os
from tep_system import TEPSentinelSystem

def calibrate():
    print("----------------------------------------------------------------")
    print("  TEP-Sentinel Threshold Calibrator")
    print("----------------------------------------------------------------")
    
    # Initialize System
    system = TEPSentinelSystem()
    
    print("\nRunning warm-up (50 steps)...")
    # Warmup is handled internally by step() returning 'WARMUP' status but we need to run it.
    for _ in range(50):
        system.step()
        
    print("Running Calibration (200 steps)...")
    mse_values = []
    
    for i in range(200):
        # Default action (steady state)
        outputs = system.step()
        # We need the RAW MSE from the autoencoder, even if system says 'NORMAL'
        # tep_system.monitor() returns is_fault, mse
        # But step() returns results. Let's rely on outputs['mse']
        
        mse = outputs['mse']
        mse_values.append(mse)
        
        if i % 20 == 0:
            print(f"Step {i}: MSE={mse:.4f}")

    mse_values = np.array(mse_values)
    min_mse = np.min(mse_values)
    max_mse = np.max(mse_values)
    mean_mse = np.mean(mse_values)
    
    # Calculate new threshold (e.g., Max observed + 20% margin or 99th percentile)
    # Using specific percentile might be safer if there are outliers
    new_threshold = np.percentile(mse_values, 99) * 1.5 # Add safety margin
    
    print("\n----------------------------------------------------------------")
    print(f"Optimization Results:")
    print(f"Min MSE: {min_mse:.4f}")
    print(f"Max MSE: {max_mse:.4f}")
    print(f"Mean MSE: {mean_mse:.4f}")
    print(f"Current Threshold: {system.threshold:.4f}")
    print(f"Proposed Threshold: {new_threshold:.4f}")
    print("----------------------------------------------------------------")
    
    # Save
    save_path = "models/tep_threshold.pkl"
    print(f"Saving new threshold to {save_path}...")
    joblib.dump(new_threshold, save_path)
    print("Done. Restart the Dashboard to apply.")

if __name__ == "__main__":
    calibrate()
