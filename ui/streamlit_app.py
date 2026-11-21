"""
Streamlit entry point: `streamlit run ui/streamlit_app.py`.

Two pages (st.navigation): the Presales chat (default) and the admin Eval & Quality Dashboard.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Cloud Presales Assistant", layout="wide")
st.navigation([
    st.Page("chat_page.py", title="Chat Presales", icon="", default=True),
    st.Page("dashboard_page.py", title="Eval & Quality Dashboard"),
]).run()
