import streamlit as st
import pandas as pd
import os
import joblib
from pathlib import Path
import traceback

def show_inference_page():
    st.header("🔮 Inference & Predictions")
    st.markdown("Use a trained model to make predictions on new data.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📦 Load Model")
        
        output_dir = st.text_input(
            "Output directory path:",
            value="eda_output",
            key="infer_output_dir"
        )
        
        pipeline_path = os.path.join(output_dir, "full_pipeline.pkl")
        
        if os.path.exists(pipeline_path):
            st.success(f"✅ Pipeline found")
            
            try:
                pipeline = joblib.load(pipeline_path)
                st.session_state.pipeline = pipeline
                
                st.markdown("#### Model Info")
                st.write(f"**Model:** {pipeline.get('model_name', 'Unknown')}")
                st.write(f"**Task:** {'Classification' if pipeline.get('is_classification') else 'Regression'}")
                
            except Exception as e:
                st.error(f"❌ Error loading pipeline: {e}")
        else:
            st.warning(f"⚠️ Pipeline not found at {pipeline_path}")
            st.info("Train a model first on the Training page")
    
    with col2:
        st.subheader("📊 New Data")
        
        new_file = st.file_uploader(
            "Upload new data for predictions",
            type=["csv", "xlsx", "xls", "parquet", "json"],
            key="inference_file"
        )
        
        if new_file and hasattr(st.session_state, 'pipeline'):
            
            new_data_path = f"temp_inference_{new_file.name}"
            with open(new_data_path, "wb") as f:
                f.write(new_file.getvalue())
            
            try:
                ext = Path(new_file.name).suffix.lower()
                if ext == ".csv":
                    new_df = pd.read_csv(new_data_path)
                elif ext in [".xlsx", ".xls"]:
                    new_df = pd.read_excel(new_data_path)
                elif ext == ".parquet":
                    new_df = pd.read_parquet(new_data_path)
                elif ext == ".json":
                    new_df = pd.read_json(new_data_path)
                
                st.markdown("#### Data Preview")
                st.dataframe(new_df.head(5), use_container_width=True)
                st.write(f"Shape: {new_df.shape[0]} rows × {new_df.shape[1]} columns")
                
                if st.button("🚀 Generate Predictions", key="predict_btn", use_container_width=True):
                    
                    with st.spinner("Making predictions..."):
                        try:
                            pipeline = st.session_state.pipeline
                            preprocessor = pipeline["preprocessor"]
                            selector = pipeline.get("selector")
                            model = pipeline["model"]
                            target_le = pipeline.get("target_le")
                            is_classification = pipeline["is_classification"]
                            
                            X_t = preprocessor.transform(new_df)
                            if hasattr(X_t, "toarray"):
                                X_t = X_t.toarray()
                            
                            if selector:
                                X_t = selector.transform(X_t)
                            
                            preds = model.predict(X_t)
                            
                            result_df = new_df.copy()
                            
                            if is_classification and target_le is not None:
                                result_df["Prediction"] = target_le.inverse_transform(preds)
                            else:
                                result_df["Prediction"] = preds
                            
                            if is_classification and hasattr(model, "predict_proba"):
                                probas = model.predict_proba(X_t)
                                if probas.shape[1] == 2:
                                    result_df["Probability_Class_1"] = probas[:, 1]
                                else:
                                    for i in range(probas.shape[1]):
                                        result_df[f"Probability_Class_{i}"] = probas[:, i]
                            
                            st.session_state.predictions = result_df
                            
                            st.markdown("---")
                            st.subheader("📋 Predictions")
                            st.dataframe(result_df, use_container_width=True)
                            
                            csv_data = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions (CSV)",
                                data=csv_data,
                                file_name=f"predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            st.success(f"✅ Generated {len(result_df)} predictions")
                        
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")
                            st.error(f"```\n{traceback.format_exc()}\n```")
                
                os.remove(new_data_path)
            
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                