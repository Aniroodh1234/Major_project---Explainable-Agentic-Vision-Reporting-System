import streamlit as st
import os
import requests
from dotenv import load_dotenv
from components.sidebar import render_sidebar

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000/api/v1")

st.set_page_config(page_title="Upload Scan", page_icon="📤", layout="centered")

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_sidebar()

st.title("📤 Upload Medical Scan")
st.markdown("Upload a medical image to run the complete MedVision AI inference pipeline.")

uploaded_file = st.file_uploader("Choose a medical image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Preview", width=300)
    
    if st.button("Run AI Analysis"):
        with st.spinner("Initializing Agent 6 (Vision Analysis) and Agent 7 (Report Generation)... This may take a few seconds."):
            try:
                # Prepare the file for the POST request
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Make the API call to FastAPI backend
                response = requests.post(f"{BACKEND_URL}/analyze", files=files)
                
                if response.status_code == 200:
                    st.success("Analysis complete!")
                    st.session_state["analysis_results"] = response.json()
                    st.switch_page("pages/Results.py")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Backend communication failed: {e}")
                st.info("Make sure the FastAPI backend is running!")
