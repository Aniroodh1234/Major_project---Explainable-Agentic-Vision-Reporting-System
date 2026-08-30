import streamlit as st
import os
from components.sidebar import render_sidebar

# Set page configuration
st.set_page_config(
    page_title="MedVision AI | Home",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_sidebar()

st.title("🧬 Welcome to MedVision AI")
st.markdown("### The future of agentic medical image analysis.")

st.markdown("""
MedVision AI is a highly advanced, multi-agent AI system designed to analyze medical imagery, predict conditions, generate clinical reports, and automatically audit its own outputs to eliminate AI hallucinations.

### 🌟 Key Features
- **Vision Transformer Inference:** Accurately classifies conditions from medical scans.
- **Explainable AI (Grad-CAM):** Visually highlights the regions in the image that led to the prediction.
- **Clinical Report Generation:** Drafts professional medical reports summarizing the findings.
- **LLM-as-a-Judge Audit:** A dedicated agent reviews and refines reports for strict clinical accuracy before they ever reach the user.

### 📤 Getting Started
To begin your analysis, simply head over to the **Upload Scan** page using the sidebar on the left.
You can upload any standard medical image in `.png`, `.jpg`, or `.jpeg` formats.
""")

st.info("👈 Navigate to the Upload page to start the analysis pipeline.")
