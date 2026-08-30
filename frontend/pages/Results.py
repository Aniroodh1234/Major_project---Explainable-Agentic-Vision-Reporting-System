import streamlit as st
import os
from components.sidebar import render_sidebar
from components.image_viewer import render_image_viewer
from components.heatmap_viewer import render_heatmap_viewer
from components.report_viewer import render_report_viewer

st.set_page_config(page_title="Analysis Results", page_icon="📊", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_sidebar()

st.title("📊 Analysis Results")

# Check if we have results in session state
if "analysis_results" not in st.session_state:
    st.warning("No analysis results found. Please upload an image first.")
    st.page_link("pages/Upload.py", label="Go to Upload Page", icon="📤")
else:
    results = st.session_state["analysis_results"]
    
    # 1. Top Level Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted Classification", value=results.get("predicted_class", "Unknown").replace("_", " ").title())
    
    with col2:
        conf = results.get("confidence_score", 0.0)
        st.metric(label="Confidence Score", value=f"{conf * 100:.2f}%")
        
    with col3:
        status = results.get("prediction_status", "UNKNOWN")
        if status == "VALID":
            st.markdown(f'<div class="badge-valid">Status: {status}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-warning">Status: {status}</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # 2. Warning Messages (if any)
    warning = results.get("warning_message")
    if warning:
        st.warning(warning)
        st.markdown("---")

    # 3. Visuals (Original vs Heatmap)
    st.markdown("### 👁️ Visual Findings")
    
    # Display original image centered and smaller
    st.markdown("#### Original Upload")
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        render_image_viewer(results.get("image_name", ""))
    
    st.markdown("---")
    
    # Display the full-width Grad-CAM visualization
    st.markdown("#### Model Explainability (Grad-CAM)")
    heatmap_path = results.get("heatmap_path", "")
    render_heatmap_viewer(heatmap_path)
        
    st.markdown("---")
    
    # 4. Clinical Report
    render_report_viewer(results.get("generated_report", ""))
    
    st.markdown("---")
    
    # 5. Agent 8 Quality Audit Details
    st.markdown("### 🤖 Agent 8 Quality Audit")
    
    audit_col1, audit_col2 = st.columns(2)
    with audit_col1:
        st.metric(label="Evaluation Score", value=f"{results.get('evaluation_score', 0.0)}%")
        
    with audit_col2:
        val_status = results.get("validation_status", "UNKNOWN")
        st.metric(label="Validation Status", value=val_status)
        
    iters = results.get("total_refinement_iterations", 1)
    st.caption(f"The report was refined {iters} time(s) by Agent 8 to achieve this quality score.")
    
    feedback = results.get("evaluation_feedback")
    if feedback and val_status != "VALIDATED":
        st.info(f"**Agent 8 Feedback:** {feedback}")
