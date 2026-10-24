# TEP-Sentinel

Fault detection, diagnosis and explanation for the Tennessee Eastman Process (TEP), with a small reinforcement-learning control demo.

![Architecture](Arquitetura%20png.png)

## What it does

| Component | Implementation | Validated on |
|---|---|---|
| **Detection** | LSTM autoencoder on 100-sample windows of the 52 variables; debounced alarm (3 consecutive exceedances) | Held-out TEP test runs (real data) |
| **Diagnosis** | CNN-LSTM classifier, 21 classes (normal + IDV 1-20), probabilities smoothed over the alarm | Held-out TEP test runs (real data) |
| **Explanation** | Gemini + ChromaDB RAG, fed with the fault catalogue and the sensors that deviated at alarm time; runs in a background thread | Not evaluated (needs an API key) |
| **Control** | PPO agent on a neural surrogate of the plant (disturbance rejection around the nominal state) | **Surrogate only**, not the real plant |

### Results (held-out test set, 10 runs per fault, `python evaluate.py`)

* 17 of 20 faults are detected in 100% of runs, with a median delay between 1 and ~90 samples (3 min per sample). Diagnosis is correct in 90-100% of alarmed runs once the smoothed output settles (read 50 samples after the alarm).
* **IDV 3, 9 and 15 are not detectable** (≈ false-alarm level). This is a known property of the benchmark.
* About 10% of fault-free 960-sample runs raise one confirmed false alarm. The threshold trades this off against the recall of subtle faults (IDV 10, 16).
* Full table: [`verification_output/evaluation.md`](verification_output/evaluation.md).

### Limitations

* The RL agent is trained and evaluated only on the neural surrogate: PPO reaches a final measurement MSE of 0.62 against 3.22 for doing nothing (`models/rl_report.json`). The surrogate learns correlations from closed-loop data, so this says nothing about control of the real process and it must not be used to act on a plant.
* The RAG explanation quality has not been measured.
* The dashboard's "synthetic overlay" adds sensor offsets for demos. It is not a physical fault model; use the replayed IDV runs to judge the detector.

## Setup

```bash
git clone https://github.com/LucasOliverno/TEP-Sentinel.git
cd TEP-Sentinel
python -m venv .venv && .venv\Scripts\activate      # Windows (use source .venv/bin/activate elsewhere)
pip install -r requirements.txt
```

Trained models (`models/`) are in the repository, so the dashboard can run as soon as the **test** data is available:

```bash
python scripts/download_data.py --only test      # ~900 MB download, converted to Parquet in data/
python launch_dashboard.py                       # or run_dashboard.bat
```

For the explanation panel, create `.env` with `GOOGLE_API_KEY=...` (optional `GEMINI_MODEL`) and build the index once: `python rag_indexer.py`. Without it the rest of the dashboard works and shows "RAG unavailable".

## Reproducing the models

```bash
python scripts/download_data.py        # all four files (~1.4 GB download, several GB of RAM while converting)
python data_processor.py               # scaler + windows in processed_data/
python train_autoencoder.py
python calibrate_threshold.py          # alarm threshold on held-out normal runs
python train_classifier.py
python train_surrogate.py
python train_rl.py --timesteps 100000
python evaluate.py
```

Training uses fixed seeds. Splits are by `simulationRun`, and windows never cross run boundaries. Samples before the fault onset (20 in training, 160 in test) are labelled normal.

## Dashboard

* **Replay**: stream any run of the test set (any of the 20 faults, or normal operation). The panel shows the anomaly score and threshold, the variables that deviated most, the classifier's top-3 and the RAG explanation.
* **Surrogate**: closed-loop simulation on the neural plant model; toggle "RL control" to let PPO act on the valves.

## Project layout

| Path | Purpose |
|---|---|
| `config.py` | Paths, dimensions, variable names, fault catalogue |
| `data_processor.py`, `scripts/download_data.py` | Dataset download and windowing |
| `train_*.py`, `calibrate_threshold.py`, `evaluate.py` | Training, calibration, evaluation |
| `fdd.py` | Models container, alarm debounce, diagnosis smoothing, top deviations |
| `sources.py` | Real-data replay and surrogate plant sources |
| `tep_system.py` | Orchestrator (source → FDD → RAG / RL) |
| `rag_agent.py`, `rag_indexer.py` | RAG agent and index builder |
| `envs/tep_env.py` | Gymnasium environment on the surrogate |
| `dashboard.py` | Streamlit UI |
| `tests/` | `pytest` suite (CI runs `ruff` and `pytest`) |
| `Banco de Conhecimento/` | Domain knowledge (indexed by the RAG) and project reports (Portuguese) |

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```
