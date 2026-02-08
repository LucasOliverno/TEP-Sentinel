import sys
import os
import subprocess
import time

def main():
    print("----------------------------------------------------------------")
    print("  TEP-Sentinel Launcher")
    print("----------------------------------------------------------------")
    print(f"Python Executable: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}")
    
    # 1. Verify Imports
    print("\n[1/3] Verifying TensorFlow...")
    try:
        import tensorflow as tf
        print(f"SUCCESS: TensorFlow {tf.__version__} found at {tf.__file__}")
    except ImportError as e:
        print(f"CRITICAL ERROR: TensorFlow not found in this environment!")
        print(f"Please install requirements: pip install -r requirements.txt")
        input("Press Enter to exit...")
        sys.exit(1)

    print("\n[2/3] Verifying Streamlit...")
    try:
        import streamlit
        print(f"SUCCESS: Streamlit {streamlit.__version__} found.")
    except ImportError as e:
        print(f"CRITICAL ERROR: Streamlit not found!")
        input("Press Enter to exit...")
        sys.exit(1)

    # 2. Launch Dashboard
    print("\n[3/3] Launching Dashboard...")
    dashboard_path = os.path.join(os.getcwd(), "dashboard.py")
    
    # Construct command: [python] -m streamlit run [dashboard.py]
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    
    print(f"Running command: {' '.join(cmd)}")
    print("----------------------------------------------------------------")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    except subprocess.CalledProcessError as e:
        print(f"\nError running dashboard: {e}")
        input("Press Enter to close...")

if __name__ == "__main__":
    main()
