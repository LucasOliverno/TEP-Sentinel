"""Calibrate the alarm threshold on HELD-OUT fault-free training runs, at alarm level.

A per-window percentile is too optimistic for a monitor that scores every sample: consecutive
windows are highly correlated. This script streams the held-out normal runs window by window,
applies the same debounce as the live system and picks the smallest threshold whose number of
confirmed false-alarm episodes stays under a budget.
"""
import argparse

import joblib
import numpy as np

from common import log, setup_logging
from config import FEATURE_COLS, THRESHOLD_PATH, WINDOW_SIZE, data_path
from data_processor import VAL_RUN_MOD, iter_runs
from evaluate import score_mse
from fdd import Models


def count_episodes(mse: np.ndarray, threshold: float, confirm: int) -> int:
    """Number of alarm episodes: streaks of >= confirm consecutive exceedances."""
    flags = np.concatenate([[False], mse > threshold, [False]]).astype(np.int8)
    starts, ends = np.where(np.diff(flags) == 1)[0], np.where(np.diff(flags) == -1)[0]
    return int(np.sum((ends - starts) >= confirm))


def calibrate(max_episodes_per_1000_runs_samples: float, confirm: int, stride: int):
    models = Models.load(load_rl=False)
    per_run, total_samples = [], 0
    for _, run, g in iter_runs(data_path("train", "normal")):
        if run % VAL_RUN_MOD:
            continue
        data = models.scaler.transform(g[FEATURE_COLS].to_numpy(np.float64)).astype(np.float32)
        per_run.append(score_mse(models, data, stride)[1])
        total_samples += len(data) - WINDOW_SIZE + 1
    budget = max_episodes_per_1000_runs_samples * total_samples / 1000.0
    all_mse = np.concatenate(per_run)
    for thr in np.unique(np.quantile(all_mse, np.linspace(0.90, 0.99999, 400))):
        episodes = sum(count_episodes(m, thr, max(1, confirm // stride)) for m in per_run)
        if episodes <= budget:
            log.info("held-out runs=%d samples=%d budget=%.2f episodes -> threshold %.4f (%d episodes)",
                     len(per_run), total_samples, budget, thr, episodes)
            return float(thr)
    raise RuntimeError("no threshold satisfies the budget")


if __name__ == "__main__":
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.05,
                    help="allowed false-alarm episodes per 1000 monitored samples (default 0.05 ~ 1 per 20k)")
    ap.add_argument("--confirm", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    thr = calibrate(a.budget, a.confirm, a.stride)
    joblib.dump(thr, THRESHOLD_PATH)
    log.info("saved %s", THRESHOLD_PATH)
