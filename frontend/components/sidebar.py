import streamlit as st

def render_sidebar():
    """Renders the navigation sidebar and project info."""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=60)
        st.title("MedVision AI")
        st.caption("Agentic Medical Imaging")
        
        st.markdown("---")
        
        st.page_link("Home.py", label="Home", icon="🏠")
        st.page_link("pages/Upload.py", label="Upload Scan", icon="📤")
        st.page_link("pages/Results.py", label="Analysis Results", icon="📊")
        st.page_link("pages/About.py", label="About", icon="ℹ️")
