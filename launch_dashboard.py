"""Launch the Streamlit dashboard with the current Python interpreter."""
import importlib.util
import subprocess
import sys
from pathlib import Path

REQUIRED = ["tensorflow", "streamlit", "stable_baselines3"]


def main():
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"Missing packages: {', '.join(missing)}. Run: {sys.executable} -m pip install -r requirements.txt")
    dashboard = Path(__file__).resolve().parent / "dashboard.py"
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard)], check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        sys.exit(f"Dashboard exited with error: {e}")


if __name__ == "__main__":
    main()
