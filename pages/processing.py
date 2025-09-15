import streamlit as st
import os
import time
from pipeline import run_emotion_pipeline_for_files

# --- Progress UI ---
def show_progress_ui():
    """
    Returns a callback to update a styled progress bar with step info.
    """
    # Create the main progress bar and status placeholders
    progress_bar = st.progress(0)
    step_text = st.empty()
    percent_text = st.empty()

    # Internal callback function
    def progress_callback(progress: float, step_name: str = ""):
        p = min(max(progress, 0.0), 1.0)
        progress_bar.progress(p)
        percent_text.markdown(f"**{p*100:.0f}% Complete**")
        if step_name:
            step_text.markdown(f"**Step:** {step_name}")
        # heartbeat to keep UI responsive
        time.sleep(0.05)

    return progress_callback


# --- Main Processing Page ---
def show():
    st.title("📊 Processing Selected Files")

    # Check if there are files to process
    if "pending_files" not in st.session_state or not st.session_state.pending_files:
        st.warning("No files selected. Returning to Welcome page.")
        st.session_state.page = "Welcome"
        st.rerun()

    # Initialize the progress callback
    cb = show_progress_ui()

    if st.session_state.get("processing", True):
        try:
            cb(0.0, "Initializing...")
            names = st.session_state.pending_files

            # Run the pipeline and provide the progress callback
            run_emotion_pipeline_for_files(names, progress_cb=cb)

            # Update session state after completion
            st.session_state.processing = False
            st.session_state.pending_files = []
            st.session_state.page = "Results"

            cb(1.0, "✅ Processing Complete")
            st.success("Processing complete!")
            st.rerun()

        except Exception as e:
            st.session_state.processing = False
            st.error(f"Processing failed: {e}")
            if st.button("⬅️ Back to Start"):
                st.session_state.page = "Welcome"
                st.rerun()
    else:
        st.info("Processing has not started.")
