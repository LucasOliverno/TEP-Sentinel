import pandas as pd
import numpy as np
import sys

def analyze_columns(file_path):
    output_file = "analysis_results_utf8.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        # Redirect stdout to file
        original_stdout = sys.stdout
        sys.stdout = f
        
        print(f"--- Analyzing {file_path} ---")
        try:
            df = pd.read_parquet(file_path)
            
            xmeas_cols = [c for c in df.columns if 'xmeas' in c.lower()]
            xmv_cols = [c for c in df.columns if 'xmv' in c.lower()]
            other_cols = [c for c in df.columns if c not in xmeas_cols and c not in xmv_cols]
            
            print(f"XMEAS columns ({len(xmeas_cols)}): {xmeas_cols[:3]} ... {xmeas_cols[-3:]}")
            print(f"XMV columns ({len(xmv_cols)}): {xmv_cols[:3]} ... {xmv_cols[-3:]}")
            print(f"Other columns: {other_cols}")
            
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)
            
            stats = df.describe().T[['min', 'max', 'mean', 'std']]
            print("\n--- Statistics Snapshot (All XMEAS) ---")
            print(stats.loc[xmeas_cols])
            
            print("\n--- Statistics Snapshot (All XMV) ---")
            print(stats.loc[xmv_cols])
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            sys.stdout = original_stdout
            
    print(f"Analysis written to {output_file}")

if __name__ == "__main__":
    analyze_columns("TEP_FaultFree_Training_fault_free_training.parquet")
