import streamlit as st
import os

def render_image_viewer(image_name: str, caption: str = "Uploaded Medical Image"):
    """
    Renders the uploaded original image.
    In a real system, the frontend would receive the image path or URL from the backend.
    For this local demo, we'll construct the path to the backend's upload directory.
    """
    backend_uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "uploads")
    image_path = os.path.join(backend_uploads_dir, image_name)
    
    if os.path.exists(image_path):
        st.image(image_path, caption=caption, width=400)
    else:
        st.error(f"Image not found at path: {image_path}")
