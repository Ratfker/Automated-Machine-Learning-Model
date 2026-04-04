import streamlit as st
import os
import sys
import json
import time
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

# ─── PAGE CONFIG ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Pipeline v2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM STYLING ───────────────────────────────────────────────────
st.markdown("""
<style>
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .status-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE INITIALIZATION ─────��───────────────────────────────
if "pipeline_trained" not in st.session_state:
    st.session_state.pipeline_trained = False
    st.session_state.best_model = None
    st.session_state.best_name = None
    st.session_state.results_df = None
    st.session_state.metrics = None
    st.session_state.output_dir = "eda_output"
    st.session_state.training_logs = []
    st.session_state.uploaded_file = None
    st.session_state.target_col = None
    st.session_state.df = None

# ─── HEADER ───────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1>🚀 AutoML Pipeline v2</h1>
    <p style="font-size: 1.1rem; color: #666;">Automated Model Builder By AYUSH YADAV</p>
    <hr style="margin: 2rem 0;">
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR NAVIGATION ───────────────────────────────────────────────
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Training", "🔮 Inference", "📈 Results", "⚙️ Settings"],
    key="page_selector"
)

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📁 Current Session")
    st.write(f"**Output Dir:** `{st.session_state.output_dir}`")
    st.write(f"**Model Trained:** {'✅ Yes' if st.session_state.pipeline_trained else '❌ No'}")
    if st.session_state.pipeline_trained:
        st.write(f"**Best Model:** {st.session_state.best_name}")

# ─── ROUTE TO PAGES ───────────────────────────────────────────────────
if page == "🏠 Training":
    from pages import training
    training.show_training_page()

elif page == "🔮 Inference":
    from pages import inference
    inference.show_inference_page()

elif page == "📈 Results":
    from pages import results
    results.show_results_page()

elif page == "⚙️ Settings":
    from pages import settings
    settings.show_settings_page()
    
