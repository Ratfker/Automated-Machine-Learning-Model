import streamlit as st
import pandas as pd
import os
from PIL import Image

def show_results_page():
    st.header("📈 Results & Analysis")
    st.markdown("View training results, metrics, and visualizations.")
    
    if not st.session_state.pipeline_trained:
        st.warning("⚠️ No trained model available")
        st.info("Train a model first on the Training page")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Metrics",
        "📈 Visualizations",
        "📋 Logs",
        "💾 Artifacts"
    ])
    
    # ─── TAB 1: METRICS ───────────────────────────────────────────────
    with tab1:
        st.subheader("Model Performance Metrics")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Best Model", st.session_state.best_name)
        
        with col_m2:
            if st.session_state.metrics:
                if "accuracy" in st.session_state.metrics:
                    st.metric(
                        "Accuracy",
                        f"{st.session_state.metrics['accuracy']:.4f}"
                    )
                elif "r2" in st.session_state.metrics:
                    st.metric(
                        "R² Score",
                        f"{st.session_state.metrics['r2']:.4f}"
                    )
        
        with col_m3:
            st.metric("Status", "✅ Complete")
        
        if st.session_state.metrics:
            st.markdown("#### Detailed Results")
            metrics_df = pd.DataFrame([st.session_state.metrics])
            st.dataframe(metrics_df, use_container_width=True)
    
    # ─── TAB 2: VISUALIZATIONS ────────────────────────────────────────
    with tab2:
        st.subheader("Pipeline Visualizations")
        
        output_dir = st.session_state.output_dir
        
        plot_files = {
            "Target Distribution": "target_distribution.png",
            "Correlation Heatmap": "correlation_heatmap.png",
            "Feature Distributions": "feature_distributions.png",
            "Confusion Matrix": "confusion_matrix.png",
            "ROC Curve": "roc_curve.png",
            "Precision-Recall": "pr_curve.png",
            "Feature Importance": "feature_importance.png",
            "Actual vs Predicted": "actual_vs_predicted.png",
            "Residual Plot": "residual_plot.png",
            "SHAP Summary": "shap_summary.png",
        }
        
        col_plot1, col_plot2 = st.columns(2)
        
        for idx, (name, filename) in enumerate(plot_files.items()):
            plot_path = os.path.join(output_dir, filename)
            
            if os.path.exists(plot_path):
                col = col_plot1 if idx % 2 == 0 else col_plot2
                with col:
                    with st.expander(f"📊 {name}", expanded=False):
                        try:
                            img = Image.open(plot_path)
                            st.image(img, use_column_width=True)
                            st.caption(f"Source: {filename}")
                        except Exception as e:
                            st.error(f"Could not load {filename}: {e}")
    
    # ─── TAB 3: LOGS ──────────────────────────────────────────────────
    with tab3:
        st.subheader("Training Logs")
        
        if st.session_state.training_logs:
            log_text = "\n".join(st.session_state.training_logs)
            st.code(log_text, language="bash")
            
            st.download_button(
                label="📥 Download Logs",
                data=log_text,
                file_name=f"training_logs.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("No logs available")
    
    # ─── TAB 4: ARTIFACTS ─────────────────────────────────────────────
    with tab4:
        st.subheader("Model & Pipeline Artifacts")
        
        output_dir = st.session_state.output_dir
        
        model_path = os.path.join(output_dir, "best_model.pkl")
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024**2)
            st.write(f"📦 **Best Model** ({file_size:.2f} MB)")
            
            with open(model_path, "rb") as f:
                st.download_button(
                    label="📥 Download Model",
                    data=f.read(),
                    file_name="best_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        pipeline_path = os.path.join(output_dir, "full_pipeline.pkl")
        if os.path.exists(pipeline_path):
            file_size = os.path.getsize(pipeline_path) / (1024**2)
            st.write(f"🔗 **Full Pipeline** ({file_size:.2f} MB)")
            
            with open(pipeline_path, "rb") as f:
                st.download_button(
                    label="📥 Download Pipeline",
                    data=f.read(),
                    file_name="full_pipeline.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
                