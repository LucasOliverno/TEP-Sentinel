"""Central configuration: paths, dimensions and process metadata for TEP-Sentinel."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("TEP_DATA_DIR", ROOT / "data"))
PROCESSED_DIR = Path(os.getenv("TEP_PROCESSED_DIR", ROOT / "processed_data"))
MODELS_DIR = ROOT / "models"
KB_DIR = ROOT / "Banco de Conhecimento"
CHROMA_DIR = ROOT / "chroma_db"

SCALER_PATH = MODELS_DIR / "tep_scaler.pkl"
AUTOENCODER_PATH = MODELS_DIR / "tep_autoencoder.keras"
CLASSIFIER_PATH = MODELS_DIR / "tep_classifier.keras"
SURROGATE_PATH = MODELS_DIR / "tep_surrogate.keras"
THRESHOLD_PATH = MODELS_DIR / "tep_threshold.pkl"
RL_BEST_PATH = MODELS_DIR / "PPO" / "best_model.zip"

WINDOW_SIZE = 100
N_XMEAS = 41
N_XMV = 11
N_FEATURES = N_XMEAS + N_XMV
NUM_CLASSES = 21  # 0 = normal, 1..20 = IDV(1..20)
RANDOM_SEED = 42

# Rieth et al. (2017): faults are introduced after 1 h (train) / 8 h (test) at 3 min sampling.
FAULT_ONSET = {"train": 20, "test": 160}

# Parquet files produced by scripts/download_data.py
DATA_FILES = {
    ("train", "normal"): "TEP_FaultFree_Training_fault_free_training.parquet",
    ("train", "faulty"): "TEP_Faulty_Training_faulty_training.parquet",
    ("test", "normal"): "TEP_FaultFree_Testing_fault_free_testing.parquet",
    ("test", "faulty"): "TEP_Faulty_Testing_faulty_testing.parquet",
}

FEATURE_COLS = [f"xmeas_{i}" for i in range(1, N_XMEAS + 1)] + [f"xmv_{i}" for i in range(1, N_XMV + 1)]


def data_path(split: str, kind: str) -> Path:
    """Resolve a dataset file, falling back to the repo root (legacy layout)."""
    name = DATA_FILES[(split, kind)]
    primary = DATA_DIR / name
    return primary if primary.exists() else ROOT / name


XMEAS_NAMES = [
    "A feed (stream 1)", "D feed (stream 2)", "E feed (stream 3)", "A+C feed (stream 4)",
    "Recycle flow", "Reactor feed rate", "Reactor pressure", "Reactor level",
    "Reactor temperature", "Purge rate", "Separator temperature", "Separator level",
    "Separator pressure", "Separator underflow", "Stripper level", "Stripper pressure",
    "Stripper underflow", "Stripper temperature", "Stripper steam flow", "Compressor work",
    "Reactor cooling water outlet T", "Separator cooling water outlet T",
    *[f"Reactor feed analysis {c}" for c in "ABCDEF"],
    *[f"Purge gas {c}" for c in "ABCDEFGH"],
    *[f"Product {c}" for c in "DEFGH"],
]
XMV_NAMES = [
    "D feed valve", "E feed valve", "A feed valve", "A+C feed valve", "Compressor recycle valve",
    "Purge valve", "Separator liquid valve", "Stripper liquid valve", "Stripper steam valve",
    "Reactor cooling water valve", "Condenser cooling water valve",
]
FEATURE_NAMES = [f"XMEAS({i + 1}) {n}" for i, n in enumerate(XMEAS_NAMES)] + [
    f"XMV({i + 1}) {n}" for i, n in enumerate(XMV_NAMES)
]

# Index of the measurements shown in the dashboard (0-based).
IDX_REACTOR_PRESSURE = 6
IDX_REACTOR_LEVEL = 7
IDX_PRODUCT_G_PCT = 39  # XMEAS(40): mol% of G in the product stream

# Canonical fault descriptions (Downs & Vogel, 1993). Used to ground the RAG prompt.
FAULT_DESCRIPTIONS = {
    1: "A/C feed ratio, B composition constant (stream 4) - step change",
    2: "B composition, A/C ratio constant (stream 4) - step change",
    3: "D feed temperature (stream 2) - step change (practically unobservable)",
    4: "Reactor cooling water inlet temperature - step change",
    5: "Condenser cooling water inlet temperature - step change",
    6: "A feed loss (stream 1) - step change",
    7: "C header pressure loss, reduced availability (stream 4) - step change",
    8: "A, B, C feed composition (stream 4) - random variation",
    9: "D feed temperature (stream 2) - random variation (practically unobservable)",
    10: "C feed temperature (stream 4) - random variation",
    11: "Reactor cooling water inlet temperature - random variation",
    12: "Condenser cooling water inlet temperature - random variation",
    13: "Reaction kinetics - slow drift",
    14: "Reactor cooling water valve - sticking",
    15: "Condenser cooling water valve - sticking (practically unobservable)",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
}
