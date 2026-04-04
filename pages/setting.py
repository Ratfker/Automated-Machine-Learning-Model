import streamlit as st
import json
import os
from datetime import datetime

def show_settings_page():
    st.header("⚙️ Settings & Configuration")
    st.markdown("Manage application settings and export configurations.")
    
    st.subheader("🎨 Appearance")
    col_theme1, col_theme2 = st.columns(2)
    
    with col_theme1:
        theme = st.radio("Theme:", ["Light", "Dark"], key="theme_setting")
    
    st.subheader("📁 Default Directories")
    default_output = st.text_input(
        "Default output directory:",
        value=st.session_state.output_dir,
        key="default_output_setting"
    )
    st.session_state.output_dir = default_output
    
    st.subheader("🗂️ Model Management")
    
    output_dir = st.session_state.output_dir
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        
        if files:
            st.write("**Available Files:**")
            for file in files:
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path) / 1024
                st.write(f"- `{file}` ({file_size:.2f} KB)")
    else:
        st.warning(f"Output directory not found: {output_dir}")
    
    st.subheader("💾 Export Configuration")
    
    config = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": st.session_state.output_dir,
        "pipeline_trained": st.session_state.pipeline_trained,
        "best_model": st.session_state.best_name if st.session_state.pipeline_trained else None,
    }
    
    config_json = json.dumps(config, indent=2)
    st.download_button(
        label="📥 Export Config (JSON)",
        data=config_json,
        file_name=f"automl_config.json",
        mime="application/json",
        use_container_width=True
    )
    
    st.subheader("ℹ️ Session Information")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.write(f"**Output Directory:** `{st.session_state.output_dir}`")
        st.write(f"**Model Trained:** {'✅ Yes' if st.session_state.pipeline_trained else '❌ No'}")
        if st.session_state.pipeline_trained:
            st.write(f"**Best Model:** {st.session_state.best_name}")
    
    with col_info2:
        st.write(f"**Target Column:** {st.session_state.target_col or 'Not set'}")
        st.write(f"**File Loaded:** {'✅ Yes' if st.session_state.uploaded_file else '❌ No'}")
        st.write(f"**Metrics Available:** {'✅ Yes' if st.session_state.metrics else '❌ No'}")
    
    st.markdown("---")
    st.subheader("🔄 Reset Session")
    
    if st.button("Reset All Session Data", key="reset_all_btn", use_container_width=True):
        st.session_state.pipeline_trained = False
        st.session_state.best_model = None
        st.session_state.best_name = None
        st.session_state.results_df = None
        st.session_state.metrics = None
        st.session_state.training_logs = []
        st.session_state.uploaded_file = None
        st.session_state.target_col = None
        st.session_state.df = None
        st.success("✅ Session reset")
        st.rerun()