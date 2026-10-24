"""Run-level evaluation of the FDD pipeline on the held-out TEP test set.

Reports, per fault: detection rate, detection delay (samples after onset) and diagnosis accuracy,
plus the false-alarm rate on fault-free runs and on the pre-onset part of faulty runs.
Windows are evaluated every `--stride` samples; an alarm needs `--confirm` consecutive exceedances.
"""
import argparse

import numpy as np

from common import log, save_json, setup_logging
from config import FAULT_DESCRIPTIONS, FAULT_ONSET, FEATURE_COLS, ROOT, WINDOW_SIZE, data_path
from data_processor import iter_runs
from fdd import Models

OUT_DIR = ROOT / "verification_output"


def make_windows(data: np.ndarray, stride: int):
    ends = np.arange(WINDOW_SIZE, len(data) + 1, stride)
    idx = (ends - WINDOW_SIZE)[:, None] + np.arange(WINDOW_SIZE)[None, :]
    return ends, data[idx].astype(np.float32)


def score_mse(models: Models, data: np.ndarray, stride: int):
    """Reconstruction error of every window ending at ends (every `stride` samples)."""
    ends, windows = make_windows(data, stride)
    mse = np.empty(len(ends))
    for s in range(0, len(ends), 512):
        w = windows[s:s + 512]
        mse[s:s + 512] = np.mean(np.square(w - models.autoencoder.predict(w, verbose=0)), axis=(1, 2))
    return ends, mse


def first_confirmed(flags: np.ndarray, confirm: int):
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= confirm:
            return i - confirm + 1  # index of the first exceedance of the streak
    return None


def evaluate(runs_per_fault: int, stride: int, confirm: int, smooth: int, diag_delay: int):
    models = Models.load(load_rl=False)
    thr = models.threshold
    onset = FAULT_ONSET["test"]
    per_fault, pre_flags, pre_confirmed, normal_flags, normal_confirmed = {}, [], [], [], []
    confusion = np.zeros((21, 21), int)

    for kind in ("normal", "faulty"):
        path = data_path("test", kind)
        if not path.exists():
            log.warning("missing %s - skipping", path)
            continue
        taken = {}
        for fault, run, g in iter_runs(path):
            if taken.get(fault, 0) >= runs_per_fault:
                continue
            taken[fault] = taken.get(fault, 0) + 1
            data = models.scaler.transform(g[FEATURE_COLS].to_numpy(np.float64)).astype(np.float32)
            ends, mse = score_mse(models, data, stride)
            flags = mse > thr
            if fault == 0:
                normal_flags.append(flags.mean())
                normal_confirmed.append(first_confirmed(flags, confirm) is not None)
                continue
            pre = ends - 1 < onset
            pre_flags.append(flags[pre].mean())
            pre_confirmed.append(first_confirmed(flags[pre], confirm) is not None)
            post = np.where(~pre)[0]
            hit = first_confirmed(flags[post], confirm)
            rec = per_fault.setdefault(fault, {"runs": 0, "detected": 0, "delays": [], "correct": 0,
                                               "window_tpr": []})
            rec["runs"] += 1
            rec["window_tpr"].append(flags[post].mean())
            if hit is not None:
                rec["detected"] += 1
                start = post[hit]
                rec["delays"].append(int(ends[start] - 1 - onset))
                # Diagnosis as the operator would see it once the smoothed classifier output has settled:
                # mean probabilities over the `smooth` windows ending `diag_delay` samples after the alarm.
                last = min(start + diag_delay, len(ends) - 1)
                first = max(start, last - smooth + 1)
                w = make_windows(data[ends[first] - WINDOW_SIZE:ends[last]], stride)[1]
                pred = int(np.argmax(models.classifier.predict(w, verbose=0).mean(axis=0)))
                rec["correct"] += int(pred == fault)
                confusion[fault, pred] += 1

    result = {
        "threshold": thr, "diag_delay": diag_delay, "stride": stride, "confirm": confirm,
        "runs_per_fault": runs_per_fault,
        "false_alarm_window_rate_fault_free": float(np.mean(normal_flags)) if normal_flags else None,
        "false_alarm_run_rate_fault_free": float(np.mean(normal_confirmed)) if normal_confirmed else None,
        "false_alarm_window_rate_pre_onset": float(np.mean(pre_flags)) if pre_flags else None,
        "false_alarm_run_rate_pre_onset": float(np.mean(pre_confirmed)) if pre_confirmed else None,
        "faults": {},
    }
    for f, r in sorted(per_fault.items()):
        result["faults"][f] = {
            "runs": r["runs"],
            "detected_runs": r["detected"],
            "detection_rate": r["detected"] / r["runs"],
            "window_tpr": float(np.mean(r["window_tpr"])),
            "median_delay_samples": float(np.median(r["delays"])) if r["delays"] else None,
            "diagnosis_accuracy_given_detected": r["correct"] / r["detected"] if r["detected"] else None,
        }
    return result, confusion


def to_markdown(res):
    lines = ["# FDD evaluation (held-out test set)", "",
             f"Threshold={res['threshold']:.4f} | stride={res['stride']} | confirm={res['confirm']} | "
             f"runs/fault={res['runs_per_fault']} | diagnosis read {res['diag_delay']} samples after the alarm", ""]
    if res["false_alarm_run_rate_fault_free"] is not None:
        lines += [f"- False alarms, fault-free runs: {res['false_alarm_window_rate_fault_free']:.2%} of windows, "
                  f"{res['false_alarm_run_rate_fault_free']:.2%} of runs raise a confirmed alarm"]
    if res["false_alarm_run_rate_pre_onset"] is not None:
        lines += [f"- False alarms, pre-onset part of faulty runs: {res['false_alarm_window_rate_pre_onset']:.2%} "
                  f"of windows, {res['false_alarm_run_rate_pre_onset']:.2%} of runs"]
    header = ("| IDV | Description | Detection rate | Window TPR | Median delay (samples) "
              "| Diagnosis acc. (alarmed runs) |")
    lines += ["", header, "|---|---|---|---|---|---|"]
    for f, r in res["faults"].items():
        d = "-" if r["median_delay_samples"] is None else f"{r['median_delay_samples']:.0f}"
        acc = r["diagnosis_accuracy_given_detected"]
        a = "-" if acc is None else f"{acc:.0%} ({r['detected_runs']})"
        desc = FAULT_DESCRIPTIONS[f][:48]
        lines.append(f"| {f} | {desc} | {r['detection_rate']:.0%} | {r['window_tpr']:.0%} | {d} | {a} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-per-fault", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--confirm", type=int, default=3)
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--diag-delay", type=int, default=50, help="windows after the alarm to read the diagnosis")
    a = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    res, cm = evaluate(a.runs_per_fault, a.stride, a.confirm, a.smooth, a.diag_delay)
    save_json(OUT_DIR / "evaluation.json", res)
    np.savetxt(OUT_DIR / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    (OUT_DIR / "evaluation.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
