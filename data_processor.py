"""Data pipeline: Parquet -> scaled sliding windows saved as memory-mappable .npy files.

Differences from the first version:
  * windows never cross a (faultNumber, simulationRun) boundary;
  * samples before the fault onset are labelled 0 (the process is still normal there);
  * every window keeps its simulationRun so train/validation can be split by run
    (overlapping windows from the same run must not land on both sides of the split);
  * data is streamed one fault at a time (pyarrow filters) to keep memory bounded.
"""
import argparse
import gc
import json

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from config import FAULT_ONSET, FEATURE_COLS, N_FEATURES, PROCESSED_DIR, SCALER_PATH, WINDOW_SIZE, data_path

VAL_RUN_MOD = 5  # simulationRun % 5 == 0 -> validation (20 %)


def fit_scaler(normal_parquet=None, save_path=SCALER_PATH):
    """Fit the StandardScaler ONLY on fault-free training data."""
    path = normal_parquet or data_path("train", "normal")
    print(f"Fitting scaler on {path} ...")
    df = pq.read_table(path, columns=FEATURE_COLS).to_pandas()
    scaler = StandardScaler().fit(df[FEATURE_COLS].to_numpy(dtype=np.float64))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to {save_path}")
    return scaler


def label_run(fault: int, n_samples: int, onset: int) -> np.ndarray:
    """Per-sample labels: 0 before the fault onset, `fault` afterwards (sample index is 0-based)."""
    labels = np.zeros(n_samples, dtype=np.int8)
    if fault > 0:
        labels[onset:] = fault
    return labels


def window_run(data: np.ndarray, labels: np.ndarray, window: int, stride: int):
    """Sliding windows over ONE run. Label = label of the last step in the window."""
    n = (len(data) - window) // stride + 1
    if n <= 0:
        return np.empty((0, window, data.shape[1]), np.float32), np.empty(0, np.int8)
    ends = np.arange(n) * stride + window  # exclusive end indices
    idx = (ends - window)[:, None] + np.arange(window)[None, :]
    return data[idx].astype(np.float32), labels[ends - 1]


def iter_runs(parquet_path):
    """Yield (fault, run, DataFrame) streaming one faultNumber at a time."""
    faults = sorted({int(f) for f in pq.read_table(parquet_path, columns=["faultNumber"])
                     .column("faultNumber").unique().to_pylist()})
    cols = ["faultNumber", "simulationRun", "sample"] + FEATURE_COLS
    for fault in faults:
        df = pq.read_table(parquet_path, columns=cols, filters=[("faultNumber", "==", fault)]).to_pandas()
        df.sort_values(["simulationRun", "sample"], inplace=True)
        for run, g in df.groupby("simulationRun", sort=True):
            yield fault, int(run), g
        del df
        gc.collect()


def process_split(split: str, stride: int, scaler, window: int = WINDOW_SIZE, out_dir=PROCESSED_DIR):
    """Create X.npy / y.npy / run.npy / fault.npy for a split ('train' or 'test')."""
    onset = FAULT_ONSET[split]
    out = out_dir / split
    out.mkdir(parents=True, exist_ok=True)
    files = [data_path(split, k) for k in ("normal", "faulty") if data_path(split, k).exists()]
    if not files:
        raise FileNotFoundError(f"No {split} parquet found. Run scripts/download_data.py")

    # Pass 1: count windows so the output can be pre-allocated on disk.
    total, per_file = 0, {}
    for f in files:
        meta = pq.read_table(f, columns=["faultNumber", "simulationRun"]).to_pandas()
        counts = meta.groupby(["faultNumber", "simulationRun"]).size()
        n = int(((counts - window) // stride + 1).clip(lower=0).sum())
        per_file[f], total = n, total + n
    print(f"[{split}] {total:,} windows (stride={stride}, window={window})")

    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(total, window, N_FEATURES))
    y = np.empty(total, np.int8)
    run_ids = np.empty(total, np.int16)
    fault_ids = np.empty(total, np.int8)

    pos = 0
    for f in files:
        print(f"  reading {f.name}")
        for fault, run, g in iter_runs(f):
            data = scaler.transform(g[FEATURE_COLS].to_numpy(dtype=np.float64)).astype(np.float32)
            labels = label_run(fault, len(data), onset)
            xw, yw = window_run(data, labels, window, stride)
            k = len(xw)
            X[pos:pos + k] = xw
            y[pos:pos + k] = yw
            run_ids[pos:pos + k] = run
            fault_ids[pos:pos + k] = fault
            pos += k
    assert pos == total, (pos, total)
    X.flush()
    np.save(out / "y.npy", y)
    np.save(out / "run.npy", run_ids)
    np.save(out / "fault.npy", fault_ids)
    (out / "meta.json").write_text(json.dumps(
        {"windows": total, "stride": stride, "window": window, "onset": onset, "val_run_mod": VAL_RUN_MOD}))
    print(f"[{split}] saved to {out}")


def load_split(split: str, mmap: bool = True, out_dir=PROCESSED_DIR):
    d = out_dir / split
    X = np.load(d / "X.npy", mmap_mode="r" if mmap else None)
    return X, np.load(d / "y.npy"), np.load(d / "run.npy"), np.load(d / "fault.npy")


def run_split_masks(run_ids: np.ndarray):
    """Boolean masks (train, val) split by simulationRun."""
    val = (run_ids % VAL_RUN_MOD) == 0
    return ~val, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stride", type=int, default=50)
    ap.add_argument("--test-stride", type=int, default=50)
    ap.add_argument("--refit-scaler", action="store_true")
    args = ap.parse_args()

    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() and not args.refit_scaler else fit_scaler()
    process_split("train", args.train_stride, scaler)
    process_split("test", args.test_stride, scaler)


if __name__ == "__main__":
    main()
