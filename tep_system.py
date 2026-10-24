"""TEP-Sentinel orchestrator: plant source -> FDD (detect, diagnose, explain) -> optional RL control."""
import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

from config import N_XMV, WINDOW_SIZE  # noqa: E402
from fdd import AlarmFilter, DiagnosisSmoother, Models, format_evidence, top_deviations  # noqa: E402
from rag_agent import RagService  # noqa: E402
from sources import ReplaySource, SurrogateSource, SyntheticOffsets  # noqa: E402

log = logging.getLogger("tep.system")

RAG_RETRIGGER_STEPS = 50  # min steps between reports when the diagnosis changes


class TEPSentinelSystem:
    """Two modes:

    * ``replay``    - real Rieth et al. simulation runs (any of the 20 faults) streamed sample by sample.
                      This is the mode that exercises detection / diagnosis honestly.
    * ``surrogate`` - closed loop on the neural surrogate, where the RL agent can act on the valves.
    """

    def __init__(self, mode: str = "replay", models: Models | None = None, rag: RagService | None = None,
                 fault: int = 0, run: int = 1, seed: int | None = None, confirm: int = 3, clear: int = 10):
        if mode not in ("replay", "surrogate"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.models = models or Models.load()
        self.rag = rag if rag is not None else RagService.create()
        self.threshold = self.models.threshold
        self.window_size = WINDOW_SIZE
        self.use_rl = False

        self.alarm = AlarmFilter(confirm, clear)
        self.smoother = DiagnosisSmoother()
        self.overlay = SyntheticOffsets()
        self.current_step = 0
        self._steps_since_rag = RAG_RETRIGGER_STEPS

        self.fault_status = "WARMUP"
        self.fault_code = 0
        self.probs = None
        self.last_mse = 0.0
        self.deviations = []
        self.onset_deviations = []  # snapshot at alarm start: closest to the root cause
        self.state = np.zeros(self.models.scaler.n_features_in_, dtype=np.float32)

        if mode == "replay":
            self.source = ReplaySource.from_parquet(self.models.scaler, fault, run)
            self.buffer = self.source.initial_window(self.window_size).copy()
            self.current_step = self.window_size
            self.state = self.buffer[-1]
        else:
            from envs.tep_env import TEPEnv
            self.source = SurrogateSource(TEPEnv(), seed=seed)
            self.buffer = np.tile(self.source.state, (self.window_size, 1)).astype(np.float32)

    # ---- helpers -------------------------------------------------------------------------
    @property
    def rl_available(self) -> bool:
        return self.mode == "surrogate" and self.models.rl_agent is not None

    @property
    def finished(self) -> bool:
        return bool(self.source.finished)

    @property
    def rag_agent(self):  # kept for UI compatibility
        return self.rag.agent

    def get_real_values(self, scaled_state: np.ndarray) -> np.ndarray:
        return self.models.scaler.inverse_transform(np.asarray(scaled_state).reshape(1, -1))[0]

    def inject_offsets(self, offsets: dict[int, float] | None = None, preset: str | None = None):
        """Synthetic sensor offsets (sigma units). Only meaningful as a demo overlay."""
        self.overlay = SyntheticOffsets.preset(preset) if preset else SyntheticOffsets(offsets)
        self.alarm.reset()
        self.smoother.reset()

    # ---- FDD -----------------------------------------------------------------------------
    def monitor(self):
        mse, _ = self.models.reconstruction_error(self.buffer)
        exceeded = mse > self.threshold
        was_active = self.alarm.active
        active = self.alarm.update(exceeded)
        if active and not was_active:
            self.onset_deviations = top_deviations(self.buffer, self.models.scaler, recent=3)
        if active:
            self.probs = self.smoother.update(self.models.class_probs(self.buffer))
            self.fault_code = int(np.argmax(self.probs))  # 0 -> anomaly the classifier cannot name
            self.fault_status = "FAULT"
            self.deviations = top_deviations(self.buffer, self.models.scaler)
        else:
            self.smoother.reset()
            self.fault_code, self.probs, self.deviations, self.onset_deviations = 0, None, [], []
            self.fault_status = "NORMAL"
        return active, mse

    # ---- main loop -----------------------------------------------------------------------
    def step(self, action_override=None):
        action = None
        if self.mode == "replay":
            if self.source.finished:
                return self._result(action)
            raw = self.source.step()
        else:
            if action_override is not None:
                action = np.asarray(action_override, dtype=np.float32)
            elif self.use_rl and self.rl_available:
                action, _ = self.models.rl_agent.predict(self.state, deterministic=True)
            else:
                action = np.zeros(N_XMV, dtype=np.float32)
            raw = self.source.step(action)

        self.current_step += 1
        self.state = self.overlay.apply(np.asarray(raw, dtype=np.float32))
        self.buffer = np.vstack([self.buffer[1:], self.state[None]])

        if self.current_step < self.window_size:
            self.fault_status, active, mse = "WARMUP", False, 0.0
        else:
            active, mse = self.monitor()
        self.last_mse = mse

        self._steps_since_rag += 1
        self.rag.poll()
        if active and self.fault_status == "FAULT" and self._steps_since_rag >= RAG_RETRIGGER_STEPS:
            evidence = format_evidence(self.onset_deviations or self.deviations, mse, self.threshold)
            if self.rag.request(self.fault_code, evidence):
                self._steps_since_rag = 0
        return self._result(action)

    def _result(self, action):
        return {
            "step": self.current_step,
            "state_scaled": self.state,
            "state_real": self.get_real_values(self.state),
            "status": self.fault_status,
            "fault_code": self.fault_code,
            "mse": self.last_mse,
            "action": action,
            "probs": self.probs,
            "deviations": self.deviations,
            "onset_deviations": self.onset_deviations,
            "rag_report": self.rag.report,
            "rag_pending": self.rag.pending,
            "rag_error": self.rag.error,
            "finished": self.finished,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    system = TEPSentinelSystem(mode="replay", fault=1, run=1)
    for i in range(400):
        out = system.step()
        if i % 40 == 0:
            print(f"step {out['step']}: {out['status']} fault={out['fault_code']} mse={out['mse']:.3f}")
