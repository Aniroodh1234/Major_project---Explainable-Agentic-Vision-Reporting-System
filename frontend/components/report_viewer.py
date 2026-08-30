import streamlit as st

def render_report_viewer(report_text: str):
    """
    Renders the validated medical report in a clean, readable format.
    """
    if not report_text:
        st.warning("No clinical report generated.")
        return
        
    st.markdown("### 📋 Final Clinical Report")
    st.markdown(f'<div class="report-box">{report_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
