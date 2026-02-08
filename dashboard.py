import os
import warnings

# Suppress Warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tep_system import TEPSentinelSystem

# Page Config
st.set_page_config(
    page_title="TEP-Sentinel Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize System in Session State
if 'system' not in st.session_state:
    with st.spinner("Initializing TEP System... Loading Models..."):
        st.session_state.system = TEPSentinelSystem()
        st.session_state.history = {
            'step': [],
            'kPI': [], # MSE
            'Reactor_Pressure': [], # XMEAS(7)
            'Reactor_Level': [], # XMEAS(8)
            'Product_G': [], # XMEAS(40)
            'status': []
        }
        st.session_state.running = False

# Sidebar Controls
st.sidebar.title("🎮 Control Panel")
start_btn = st.sidebar.button("▶ START Simulation")
stop_btn = st.sidebar.button("⏹ STOP")
reset_btn = st.sidebar.button("🔄 RESET")

# Fault Injection
st.sidebar.markdown("---")
st.sidebar.subheader("💉 Fault Injection")
fault_type = st.sidebar.selectbox("Select Fault Condition", ["None", "IDV(1): A/C Imbalance", "IDV(6): A Feed Loss"])
if st.sidebar.button("Apply Condition"):
    st.session_state.system.inject_fault(fault_type)
    if fault_type == "None":
        st.sidebar.success("System Restored to Normal")
    else:
        st.sidebar.warning(f"Fault {fault_type} Active!")

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False
if reset_btn:
    st.session_state.system = TEPSentinelSystem() # Re-init
    st.session_state.history = {'step': [], 'kPI': [], 'Reactor_Pressure': [], 'Reactor_Level': [], 'Product_G': [], 'status': []}

# Main Layout
st.title("🏭 TEP-Sentinel: Intelligent Control Center")
st.markdown("### Diagnosis & Control System (FDD + RL + RAG)")

# Metrics Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
status_placeholder = st.empty()
report_placeholder = st.container()
plots_placeholder = st.empty()

# Simulation Loop
if st.session_state.running:
    # Run one step
    outputs = st.session_state.system.step()
    
    # Store History (Circular Buffer for Display)
    hist = st.session_state.history
    hist['step'].append(outputs['step'])
    hist['kPI'].append(outputs['mse'])
    # Indices: XMEAS(7)=6 (0-based), XMEAS(8)=7, XMEAS(40)=39? No, checking dicionario 
    # XMEAS 1-41 -> Indices 0-40. 
    # Reactor Pressure = XMEAS 7 -> Index 6.
    # Reactor Level = XMEAS 8 -> Index 7.
    # Rate G = XMEAS 40 -> Index 39.
    
    real_vals = outputs['state_real']
    hist['Reactor_Pressure'].append(real_vals[6])
    hist['Reactor_Level'].append(real_vals[7])
    hist['Product_G'].append(real_vals[39])
    hist['status'].append(outputs['status'])
    
    # Keep last 100 points
    if len(hist['step']) > 100:
        for k in hist:
            hist[k] = hist[k][-100:]

# --- RENDER UI (Always) ---
# Get latest data (even if not running)
if len(st.session_state.history['step']) > 0:
    last_real = [
        st.session_state.history['Reactor_Pressure'][-1],
        st.session_state.history['Reactor_Level'][-1],
        st.session_state.history['Product_G'][-1]
    ]
    last_mse = st.session_state.history['kPI'][-1]
    last_status = st.session_state.history['status'][-1]
    
    # Update KPIs
    kpi1.metric("Reactor Pressure", f"{last_real[0]:.2f} kPa")
    kpi2.metric("Reactor Level", f"{last_real[1]:.2f} %")
    kpi3.metric("Production G", f"{last_real[2]:.2f} kmol/h")
    kpi4.metric("Anomalies (MSE)", f"{last_mse:.4f}")

    # Update Status
    if last_status == "FAULT":
        # Check fault code if available in system
        code = st.session_state.system.fault_code
        status_placeholder.error(f"🚨 FAULT DETECTED: Code {code}")
    else:
        status_placeholder.success("✅ SYSTEM NORMAL")

    # --- RAG AI Section ---
    rag_container = report_placeholder.container()
    
    # Check for Agent Availability
    if st.session_state.system.rag_agent is None:
        with rag_container.expander("⚠️ RAG AI Unavailable", expanded=False):
            st.warning("The RAG Agent could not be initialized.")
            if hasattr(st.session_state.system, 'rag_init_error') and st.session_state.system.rag_init_error:
                st.error(f"Error: {st.session_state.system.rag_init_error}")
            st.info("Check if GOOGLE_API_KEY is set in .env")

    # Check for Runtime Errors
    elif hasattr(st.session_state.system, 'rag_runtime_error') and st.session_state.system.rag_runtime_error:
        with rag_container.expander("⚠️ RAG Diagnosis Failed", expanded=True):
            st.error(f"Error generating report: {st.session_state.system.rag_runtime_error}")

    # Check for Successful Report
    elif st.session_state.system.last_rag_report:
        with rag_container.expander("📝 RAG AI Diagnosis Report", expanded=True):
            st.markdown(st.session_state.system.last_rag_report)
            
    # Check for Pending Analysis
    elif last_status == "FAULT":
        with rag_container.expander("⏳ Analyzing Fault...", expanded=True):
            st.info("The AI is analyzing the fault context... Please wait (3-5s).")

    # Plots
    with plots_placeholder.container():
        col1, col2 = st.columns(2)
        
        # Plot 1: Reactor Pressure
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=st.session_state.history['Reactor_Pressure'], mode='lines', name='Pressure'))
        fig1.update_layout(
            title="Reactor Pressure (XMEAS 7)", 
            height=300, 
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True),
        )
        col1.plotly_chart(fig1, key="plot_pressure")
        
        # Plot 2: MSE (Anomaly Score)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=st.session_state.history['kPI'], mode='lines', name='MSE', line=dict(color='red')))
        fig2.add_hline(y=st.session_state.system.threshold, line_dash="dash", annotation_text="Threshold")
        fig2.update_layout(
            title="Anomaly Score (MSE)", 
            height=300, 
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True),
        )
        col2.plotly_chart(fig2, key="plot_mse")

# --- LOOP CONTROL ---
if st.session_state.running:
    # Increase sleep to 0.5s to prevent flickering and reduce CPU load
    time.sleep(0.5) 
    st.rerun()

else:
    st.info("Simulation Paused. Press START to resume.")
