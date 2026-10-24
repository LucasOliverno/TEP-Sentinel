import numpy as np
import pandas as pd
import pytest

import sources
from config import FEATURE_COLS, MODELS_DIR, SCALER_PATH
from fdd import Models
from rag_agent import RagService
from tep_system import TEPSentinelSystem

pytestmark = pytest.mark.skipif(not SCALER_PATH.exists() or not (MODELS_DIR / "tep_autoencoder.keras").exists(),
                                reason="trained models not available")


@pytest.fixture(scope="module")
def models():
    return Models.load(load_rl=False)


def fake_run(scaler, n=260):
    """A run that is exactly nominal (scaled value 0) for n samples."""
    return pd.DataFrame(np.tile(scaler.mean_, (n, 1)), columns=FEATURE_COLS).assign(sample=np.arange(1, n + 1))


def test_replay_system_runs_and_flags_big_fault(models, monkeypatch):
    run = fake_run(models.scaler)
    run.loc[150:, "xmeas_7"] += 40 * np.sqrt(models.scaler.var_[6])  # huge reactor pressure excursion
    monkeypatch.setattr(sources.pq, "read_table", lambda *a, **k: type("T", (), {"to_pandas": lambda self: run})())
    monkeypatch.setattr(sources, "data_path", lambda split, kind: "dummy")
    rag = RagService(None, "disabled in tests")
    system = TEPSentinelSystem(mode="replay", models=models, rag=rag, fault=1, run=1)

    statuses = []
    for _ in range(150):
        out = system.step()
        statuses.append(out["status"])
    assert statuses[0] in ("NORMAL", "FAULT")
    assert statuses[-1] == "FAULT"
    assert out["deviations"][0]["index"] == 6
    assert out["state_real"].shape == (52,)
    assert out["rag_report"] is None


def test_rag_service_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    svc = RagService.create()
    assert not svc.available
    assert svc.request(1, "x") is False
    svc.poll()
