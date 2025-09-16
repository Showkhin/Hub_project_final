import streamlit as st
import time
import logging
from pipeline import run_emotion_pipeline_for_files

logging.basicConfig(level=logging.INFO)

def show_progress_ui():
    """Return a callback updating progress bar and text."""
    progress_bar = st.progress(0)
    step_text = st.empty()
    percent_text = st.empty()

    def progress_callback(progress: float, step_name: str = ""):
        p = min(max(progress, 0.0), 1.0)
        progress_bar.progress(p)
        percent_text.markdown(f"**{p*100:.0f}% Complete**")
        if step_name:
            step_text.markdown(f"**Step:** {step_name}")
        time.sleep(0.05)  # keep UI responsive

    return progress_callback

def show():
    st.title("📊 Processing Selected Files")

    if "pending_files" not in st.session_state or not st.session_state.pending_files:
        st.warning("No files selected. Returning to Welcome page.")
        st.session_state.page = "Welcome"
        st.rerun()

    cb = show_progress_ui()

    if st.session_state.get("processing", True):
        try:
            logging.info("Starting processing pipeline")
            cb(0.0, "Initializing...")

            names = st.session_state.pending_files
            run_emotion_pipeline_for_files(names, progress_cb=cb)


            st.session_state.processing = False
            st.session_state.pending_files = []
            st.session_state.page = "Results"

            cb(1.0, "✅ Processing Complete")
            st.success("Processing complete!")
            st.rerun()

        except Exception as e:
            logging.error(f"Processing failed: {e}")
            st.session_state.processing = False
            st.error(f"Processing failed: {e}")
            if st.button("⬅️ Back to Start"):
                st.session_state.page = "Welcome"
                st.rerun()
    else:
        st.info("Processing has not started.")
