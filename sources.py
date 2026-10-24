"""Plant data sources: real-data replay and the neural surrogate (closed loop)."""
import numpy as np
import pyarrow.parquet as pq

from config import FEATURE_COLS, N_XMV, data_path


class SyntheticOffsets:
    """Adds ramped offsets (in sigma units, scaled space) to chosen variables.

    This is a DEMO overlay for fault-free data, not a physical fault model.
    """

    PRESETS = {
        "None": {},
        "Synthetic: A feed loss (XMEAS 1 -> -8 sigma)": {0: -8.0},
    }

    def __init__(self, offsets: dict | None = None, ramp_steps: int = 10):
        self.offsets = dict(offsets or {})
        self.ramp_steps = max(1, ramp_steps)
        self._age = 0

    @classmethod
    def preset(cls, name: str):
        return cls(cls.PRESETS[name])

    @property
    def active(self) -> bool:
        return bool(self.offsets)

    def apply(self, state: np.ndarray) -> np.ndarray:
        if not self.offsets:
            return state
        self._age += 1
        gain = min(1.0, self._age / self.ramp_steps)
        out = state.copy()
        for idx, sigma in self.offsets.items():
            out[idx] += sigma * gain
        return out


def available_replays(split: str = "test"):
    """{fault_id: n_runs} for the datasets present locally (0 = fault-free)."""
    out = {}
    for kind in ("normal", "faulty"):
        p = data_path(split, kind)
        if p.exists():
            col = pq.read_table(p, columns=["faultNumber", "simulationRun"]).to_pandas()
            for f, g in col.groupby("faultNumber"):
                out[int(f)] = int(g["simulationRun"].nunique())
    return dict(sorted(out.items()))


class ReplaySource:
    """Replays one real simulation run sample by sample (already scaled)."""

    def __init__(self, scaled_run: np.ndarray, fault: int = 0, run: int = 1, onset: int | None = None):
        self.data = scaled_run.astype(np.float32)
        self.fault, self.run, self.onset = fault, run, onset
        self.pos = 0

    @classmethod
    def from_parquet(cls, scaler, fault: int, run: int, split: str = "test"):
        from config import FAULT_ONSET
        kind = "normal" if fault == 0 else "faulty"
        path = data_path(split, kind)
        tbl = pq.read_table(path, columns=["sample"] + FEATURE_COLS,
                            filters=[("faultNumber", "==", fault), ("simulationRun", "==", run)]).to_pandas()
        if tbl.empty:
            raise ValueError(f"No data for fault={fault} run={run} in {path}")
        tbl.sort_values("sample", inplace=True)
        scaled = scaler.transform(tbl[FEATURE_COLS].to_numpy(dtype=np.float64))
        return cls(scaled, fault, run, FAULT_ONSET[split] if fault else None)

    def __len__(self):
        return len(self.data)

    @property
    def finished(self) -> bool:
        return self.pos >= len(self.data)

    def initial_window(self, window: int) -> np.ndarray:
        """First `window` samples, consumed as warm-up."""
        w = self.data[:window]
        self.pos = len(w)
        return w

    def step(self) -> np.ndarray:
        if self.finished:
            raise StopIteration("Replay finished")
        x = self.data[self.pos]
        self.pos += 1
        return x


class SurrogateSource:
    """Closed-loop simulation on the neural surrogate (control demo)."""

    def __init__(self, env, seed: int | None = None):
        self.env = env
        self.state, _ = env.reset(seed=seed)
        self.finished = False

    def step(self, action: np.ndarray | None = None) -> np.ndarray:
        action = np.zeros(N_XMV, dtype=np.float32) if action is None else action
        self.state, _, terminated, truncated, _ = self.env.step(action)
        self.finished = terminated or truncated
        return self.state

