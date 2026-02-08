import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib
import gc

class TEPProcessor:
    def __init__(self, window_size=100, stride=10):
        self.window_size = window_size
        self.stride = stride
        self.scaler = StandardScaler()
        self.output_dir = "processed_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fit_scaler(self, df_normal):
        """Fits the scaler ONLY on normal data to correct data leakage."""
        print("Fitting scaler on Normal data...")
        # Select only process variables (exclude metadata)
        cols = [c for c in df_normal.columns if 'xmeas' in c or 'xmv' in c]
        self.feature_cols = cols
        
        self.scaler.fit(df_normal[cols])
        print("Scaler fitted.")
        
        # Save scaler for future inference
        joblib.dump(self.scaler, f"{self.output_dir}/tep_scaler.pkl")
        print(f"Scaler saved to {self.output_dir}/tep_scaler.pkl")
        
    def normalize(self, df):
        """Applies normalization using the fitted scaler."""
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df_scaled

    def create_windows(self, df, label_col='faultNumber'):
        """
        Creates 3D tensors (Samples, Window, Features) using pre-allocation.
        """
        data = df[self.feature_cols].values.astype(np.float32)
        labels = df[label_col].values
        
        num_samples = (len(df) - self.window_size) // self.stride + 1
        print(f"Creating {num_samples} windows (Size={self.window_size}, Stride={self.stride})...")
        
        # Pre-allocate memory
        X = np.empty((num_samples, self.window_size, len(self.feature_cols)), dtype=np.float32)
        y = np.empty((num_samples,), dtype=np.int8)
        
        for i in range(num_samples):
            start = i * self.stride
            end = start + self.window_size
            
            X[i] = data[start:end]
            y[i] = labels[end - 1] # Label of the last step
            
        return X, y

    def process_pipeline(self):
        # 1. Fit Scaler First (Needs Normal Train Data)
        print("Loading Train Normal for Scaler fitting...")
        df_train_normal = pd.read_parquet("TEP_FaultFree_Training_fault_free_training.parquet")
        self.fit_scaler(df_train_normal)
        del df_train_normal
        gc.collect()
        
        # 2. Sequential Processing
        # Configuration for each split
        # We perform file loading INSIDE the loop to save memory
        configs = [
            {
                'name': 'train',
                'files': ["TEP_FaultFree_Training_fault_free_training.parquet", 
                          "TEP_Faulty_Training_faulty_training.parquet"],
                'stride': 10
            },
            {
                'name': 'test',
                'files': ["TEP_FaultFree_Testing_fault_free_testing.parquet", 
                          "TEP_Faulty_Testing_faulty_testing.parquet"],
                'stride': 100
            }
        ]
        
        for config in configs:
            name = config['name']
            print(f"--- Processing {name} set (Stride={config['stride']}) ---")
            
            dfs = []
            for f in config['files']:
                print(f"Loading {f}...")
                dfs.append(pd.read_parquet(f))
            
            df = pd.concat(dfs)
            del dfs
            gc.collect()
            
            print("Normalizing...")
            # Normalize in-place or carefully
            df_norm = self.normalize(df)
            del df
            gc.collect()
            
            # Update stride temporarily
            original_stride = self.stride
            self.stride = config['stride']
            
            try:
                X, y = self.create_windows(df_norm)
                print(f"{name} shape: X={X.shape}, y={y.shape}")
                
                # Save
                save_path = f"{self.output_dir}/tep_{name}.npz"
                np.savez_compressed(save_path, X=X, y=y)
                print(f"Saved to {save_path}")
                
                del X, y, df_norm
                gc.collect()
                
            except MemoryError:
                print(f"CRITICAL: MemoryError processing {name}.")
            finally:
                self.stride = original_stride # Restore

if __name__ == "__main__":
    processor = TEPProcessor(window_size=100)
    processor.process_pipeline()
