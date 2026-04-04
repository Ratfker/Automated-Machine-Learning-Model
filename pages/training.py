import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import json
import time
from pathlib import Path
import joblib
import traceback

def show_training_page():
    st.header("🎯 Model Training")
    st.markdown("Upload your data and configure the pipeline to train models.")
    
    col1, col2 = st.columns([1, 2])
    
    # ─── LEFT COLUMN: FILE UPLOAD ─────────────────────────────────────
    with col1:
        st.subheader("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=["csv", "xlsx", "xls", "parquet", "json"],
            key="file_uploader"
        )
        
        if uploaded_file:
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.session_state.uploaded_file = file_path
            st.success(f"✅ File loaded: {uploaded_file.name}")
            
            try:
                ext = Path(uploaded_file.name).suffix.lower()
                if ext == ".csv":
                    df = pd.read_csv(file_path)
                elif ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(file_path)
                elif ext == ".parquet":
                    df = pd.read_parquet(file_path)
                elif ext == ".json":
                    df = pd.read_json(file_path)
                
                st.session_state.df = df
                
                st.markdown("#### 📊 Dataset Info")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col_b:
                    st.metric("Columns", df.shape[1])
                with col_c:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                with st.expander("📝 Column Info"):
                    col_info = pd.DataFrame({
                        "Column": df.columns,
                        "Type": df.dtypes.astype(str),
                        "Non-Null": df.count(),
                        "Null %": (df.isnull().sum() / len(df) * 100).round(2)
                    })
                    st.dataframe(col_info, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    # ─── RIGHT COLUMN: CONFIGURATION ─────────────────────────────────
    with col2:
        st.subheader("⚙️ Pipeline Configuration")
        
        if st.session_state.df is not None:
            
            st.markdown("#### Target Column")
            target_col = st.selectbox(
                "Select target column:",
                options=st.session_state.df.columns.tolist(),
                key="target_select"
            )
            st.session_state.target_col = target_col
            
            with st.expander("🎯 Target Analysis", expanded=True):
                target_data = st.session_state.df[target_col]
                n_unique = target_data.nunique()
                
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    st.metric("Unique Values", n_unique)
                with col_t2:
                    st.metric("Missing", target_data.isnull().sum())
                with col_t3:
                    st.metric("Data Type", str(target_data.dtype))
                
                if n_unique <= 20:
                    st.markdown("**Value Distribution:**")
                    st.bar_chart(target_data.value_counts())
            
            st.markdown("#### Task Type")
            task_type = st.radio(
                "Select task type:",
                options=["Auto-detect", "Classification", "Regression"],
                key="task_type"
            )
            
            task_type_map = {
                "Auto-detect": "auto",
                "Classification": "classification",
                "Regression": "regression"
            }
            
            st.markdown("#### Hyperparameters")
            
            col_hp1, col_hp2 = st.columns(2)
            
            with col_hp1:
                test_size = st.slider(
                    "Test Size Ratio",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    key="test_size"
                )
            
            with col_hp2:
                n_trials = st.slider(
                    "Optuna Trials per Model",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=10,
                    key="n_trials"
                )
            
            st.markdown("#### Feature Engineering")
            add_interactions = st.checkbox(
                "Add Polynomial Interaction Features",
                value=False,
                key="interactions"
            )
            
            st.markdown("#### Output Directory")
            output_dir = st.text_input(
                "Output directory (for plots, models, logs):",
                value="eda_output",
                key="output_dir"
            )
            st.session_state.output_dir_path = output_dir
            
            st.markdown("---")
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🚀 Start Training", use_container_width=True, key="train_btn"):
                    run_training(
                        file_path=st.session_state.uploaded_file,
                        target_col=target_col,
                        test_size=test_size,
                        n_trials=n_trials,
                        task_type=task_type_map[task_type],
                        add_interactions=add_interactions,
                        output_dir=output_dir
                    )
            
            with col_btn2:
                if st.button("🔄 Reset", use_container_width=True, key="reset_btn"):
                    st.session_state.pipeline_trained = False
                    st.session_state.best_model = None
                    st.session_state.training_logs = []
                    st.rerun()
        
        else:
            st.info("👆 Please upload a file to configure the pipeline")


def run_training(file_path, target_col, test_size, n_trials, task_type, add_interactions, output_dir):
    """Execute the AutoML pipeline via subprocess."""
    
    st.subheader("📊 Training Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    logs_area = st.empty()
    
    try:
        cmd = [
            "python", "improvedversion.py",
            file_path, target_col,
            "--test-size", str(test_size),
            "--trials", str(n_trials),
            "--output", output_dir,
            "--task-type", task_type
        ]
        
        if add_interactions:
            cmd.append("--interactions")
        
        status_text.info("⏳ Starting pipeline execution...")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        logs = []
        for line in process.stdout:
            logs.append(line.rstrip())
            
            with logs_area.container():
                st.code("\n".join(logs[-20:]), language="bash")
            
            if "Data Cleaning" in line:
                progress_bar.progress(10)
            elif "Exploratory Data Analysis" in line:
                progress_bar.progress(15)
            elif "Preparing Data" in line:
                progress_bar.progress(25)
            elif "Feature Selection" in line:
                progress_bar.progress(40)
            elif "Model Training" in line:
                progress_bar.progress(50)
            elif "Bayesian Hyperparameter" in line:
                progress_bar.progress(60)
            elif "Repeated K-Fold" in line:
                progress_bar.progress(75)
            elif "Model Evaluation" in line:
                progress_bar.progress(85)
            elif "Pipeline Complete" in line:
                progress_bar.progress(100)
        
        process.wait()
        
        if process.returncode == 0:
            st.session_state.training_logs = logs
            status_text.success("✅ Training completed successfully!")
            progress_bar.progress(100)
            
            try:
                pipeline_path = os.path.join(output_dir, "full_pipeline.pkl")
                if os.path.exists(pipeline_path):
                    pipeline = joblib.load(pipeline_path)
                    st.session_state.pipeline_trained = True
                    st.session_state.best_model = pipeline.get("model")
                    st.session_state.best_name = pipeline.get("model_name")
                    st.session_state.metrics = pipeline.get("metrics")
                    
                    st.markdown("---")
                    st.subheader("🏆 Best Model Results")
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("Best Model", st.session_state.best_name)
                    
                    with col_r2:
                        if isinstance(st.session_state.metrics, dict):
                            if "accuracy" in st.session_state.metrics:
                                st.metric("Accuracy", f"{st.session_state.metrics['accuracy']:.4f}")
                            elif "r2" in st.session_state.metrics:
                                st.metric("R² Score", f"{st.session_state.metrics['r2']:.4f}")
                    
                    with col_r3:
                        st.success(f"✅ Model saved to: {output_dir}/")
                
            except Exception as e:
                st.warning(f"⚠️ Could not load pipeline results: {e}")
        
        else:
            status_text.error("❌ Training failed!")
            st.error("Check logs above for details.")
    
    except Exception as e:
        status_text.error(f"❌ Error: {e}")
        st.error(f"```\n{traceback.format_exc()}\n```")
        
