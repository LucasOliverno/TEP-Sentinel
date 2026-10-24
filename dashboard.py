import os
import warnings
from datetime import timedelta

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from config import FAULT_DESCRIPTIONS, FEATURE_NAMES, IDX_PRODUCT_G_PCT, IDX_REACTOR_LEVEL, IDX_REACTOR_PRESSURE  # noqa: E402
from fdd import Models  # noqa: E402
from rag_agent import RagService  # noqa: E402
from sources import SyntheticOffsets, available_replays  # noqa: E402
from tep_system import TEPSentinelSystem  # noqa: E402

HISTORY = 300

st.set_page_config(page_title="TEP-Sentinel", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Loading models...")
def get_models() -> Models:
    return Models.load()


@st.cache_resource(show_spinner="Connecting RAG...")
def get_rag() -> RagService:
    return RagService.create()


@st.cache_data(show_spinner=False)
def get_replays():
    return available_replays("test")


def new_history():
    return {k: [] for k in ("step", "mse", "pressure", "level", "product_g", "status")}


def build_system(mode, fault, run):
    st.session_state.system = TEPSentinelSystem(mode=mode, models=get_models(), rag=get_rag(), fault=fault, run=run)
    st.session_state.history = new_history()
    st.session_state.running = False


# ---------------------------------------------------------------- sidebar
replays = get_replays()
st.sidebar.title("Control panel")
mode_label = st.sidebar.radio("Plant source", ["Replay (real TEP runs)", "Surrogate (closed loop + RL)"],
                              help="Replay streams real simulation runs from the test set. "
                                   "Surrogate is a neural model of the plant, only meaningful for the control demo.")
mode = "replay" if mode_label.startswith("Replay") else "surrogate"

fault, run = 0, 1
if mode == "replay":
    if not replays:
        st.sidebar.error("No dataset found. Run `python scripts/download_data.py`.")
        st.stop()
    fault = st.sidebar.selectbox(
        "Fault (IDV)", list(replays), format_func=lambda f: "0 - Normal operation" if f == 0
        else f"{f} - {FAULT_DESCRIPTIONS[f][:40]}")
    run = st.sidebar.number_input("Simulation run", 1, replays[fault], 1)

if "system" not in st.session_state or st.session_state.get("config") != (mode, fault, run):
    st.session_state.config = (mode, fault, run)
    build_system(mode, fault, run)

system: TEPSentinelSystem = st.session_state.system
hist = st.session_state.history

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("Start"):
    st.session_state.running = True
if c2.button("Stop"):
    st.session_state.running = False
if c3.button("Reset"):
    build_system(mode, fault, run)
    st.rerun()

steps_per_tick = st.sidebar.slider("Samples per refresh", 1, 20, 5)

st.sidebar.markdown("---")
if mode == "surrogate":
    system.use_rl = st.sidebar.toggle("RL control", value=False, disabled=not system.rl_available,
                                      help=None if system.rl_available else "Train it with `python train_rl.py`.")
st.sidebar.subheader("Synthetic overlay")
preset = st.sidebar.selectbox("Preset", list(SyntheticOffsets.PRESETS), help="Demo only: adds offsets to sensors.")
var_idx = st.sidebar.selectbox("Custom offset on", range(len(FEATURE_NAMES)), format_func=lambda i: FEATURE_NAMES[i])
sigma = st.sidebar.slider("Offset (sigma)", -10.0, 10.0, 0.0, 0.5)
if st.sidebar.button("Apply overlay"):
    if sigma != 0.0:
        system.inject_offsets({var_idx: sigma})
    else:
        system.inject_offsets(preset=preset)

# ---------------------------------------------------------------- main
st.title("TEP-Sentinel")
st.caption("Fault detection (autoencoder), diagnosis (CNN-LSTM) and explanation (RAG) - Tennessee Eastman Process")


def advance():
    for _ in range(steps_per_tick):
        out = system.step()
        if out["finished"]:
            st.session_state.running = False
            break
        real = out["state_real"]
        hist["step"].append(out["step"])
        hist["mse"].append(out["mse"])
        hist["pressure"].append(real[IDX_REACTOR_PRESSURE])
        hist["level"].append(real[IDX_REACTOR_LEVEL])
        hist["product_g"].append(real[IDX_PRODUCT_G_PCT])
        hist["status"].append(out["status"])
    for k in hist:
        del hist[k][:-HISTORY]
    return out


def line(y, title, color=None, thr=None):
    fig = go.Figure(go.Scatter(x=hist["step"], y=y, mode="lines", line=dict(color=color) if color else None))
    if thr is not None:
        fig.add_hline(y=thr, line_dash="dash", annotation_text="threshold")
    fig.update_layout(title=title, height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def panel():
    if st.session_state.running:
        advance()
    if not hist["step"]:
        st.info("Press Start to stream data.")
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reactor pressure", f"{hist['pressure'][-1]:.1f} kPa")
    k2.metric("Reactor level", f"{hist['level'][-1]:.1f} %")
    k3.metric("Product G", f"{hist['product_g'][-1]:.2f} mol %")
    k4.metric("Anomaly score (MSE)", f"{hist['mse'][-1]:.3f}", help=f"Alarm threshold: {system.threshold:.3f}")

    if system.finished:
        st.info("Replay finished.")
    if system.fault_status == "FAULT":
        if system.fault_code:
            st.error(f"FAULT: IDV({system.fault_code}) - {FAULT_DESCRIPTIONS[system.fault_code]}")
        else:
            st.error("FAULT: anomaly detected, classifier cannot name it")
    elif system.fault_status == "WARMUP":
        st.info("Warming up...")
    else:
        st.success("System normal")

    left, right = st.columns(2)
    left.plotly_chart(line(hist["pressure"], "Reactor pressure (XMEAS 7)"), key="p")
    right.plotly_chart(line(hist["mse"], "Anomaly score", "red", system.threshold), key="m")

    if system.fault_status == "FAULT":
        a, b = st.columns(2)
        devs = system.deviations
        fig = go.Figure(go.Bar(x=[d["z"] for d in devs][::-1], y=[d["variable"] for d in devs][::-1], orientation="h"))
        fig.update_layout(title="Largest deviations (sigma)", height=260, margin=dict(l=10, r=10, t=40, b=10))
        a.plotly_chart(fig, key="dev")
        if system.probs is not None:
            top = np.argsort(-system.probs)[:3]
            b.markdown("**Classifier top-3**")
            for i in top:
                name = "Normal/unknown" if i == 0 else f"IDV({i})"
                b.progress(float(system.probs[i]), text=f"{name}: {system.probs[i]:.0%}")

    rag = system.rag
    if not rag.available:
        with st.expander("RAG unavailable", expanded=False):
            st.warning(rag.init_error)
    elif rag.error:
        st.warning(f"RAG report failed: {rag.error}")
    elif rag.pending:
        st.info("Generating explanation...")
    elif rag.report:
        with st.expander(f"Explanation for IDV({rag.report_for})", expanded=True):
            st.markdown(rag.report)


# `run_every` is fixed at decoration time, so rebuild the fragment each run.
refresh = timedelta(seconds=0.5) if st.session_state.running else None
st.fragment(panel, run_every=refresh)()
