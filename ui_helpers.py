import streamlit as st
import pandas as pd
from oci_helpers import load_cloud_csv

def show_csv(filename: str, height: int = 300):
    df = load_cloud_csv(filename)
    if df.empty:
        st.info("No data found.")
    else:
        st.dataframe(df, height=height)

def confirmation_modal():
    # Simple modal using Streamlit elements
    if st.session_state.get("show_modal", False):
        st.warning("Are you sure you want to proceed? This action is irreversible.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm"):
                st.session_state["confirmed"] = True
                st.session_state["show_modal"] = False
        with col2:
            if st.button("Cancel"):
                st.session_state["confirmed"] = False
                st.session_state["show_modal"] = False
