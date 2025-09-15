import streamlit as st
from oci_helpers import list_wavs, upload_blob
from ui_helpers import confirmation_modal, show_csv  # Assuming these are in ui_helpers.py

# --- Constants for limits ---
MAX_SAMPLE_SELECTION = 10
MAX_UPLOAD_FILES = 10

# Add this at the top of welcome.py show()
st.sidebar.markdown("## Navigation")
if st.sidebar.button("⬅️ Welcome", key="btn_welcome"):
    st.session_state.page = "Welcome"
    st.rerun()

if st.sidebar.button("📊 Results", key="btn_results"):
    st.session_state.page = "Results"
    st.rerun()


def show():
    # Hero welcome message
    st.markdown("""
    <div class="app-hero">
      <h1>Welcome, <b>Lord Commander</b> ⚔️</h1>
      <p class="soft">Have a look in our <b>Trashery</b> — review incidents, pick samples, and process emotions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### 📄 final_emotion_ensemble.csv")
    show_csv("final_emotion_ensemble.csv", height=360)

    st.write("")
    st.markdown(f"### 🎧 Sample selection — choose up to {MAX_SAMPLE_SELECTION} WAV files (prefix: <code>sample</code>)")
    sample_names = list_wavs(prefix="sample")
    sel_samples = st.multiselect(
        "Choose sample files:",
        options=sample_names,
        max_selections=MAX_SAMPLE_SELECTION
    )

    left, right = st.columns([10, 1])
    with left:
        st.caption(f"{len(sample_names)} available in cloud bucket.")
    with right:
        if st.button("Proceed", key="proceed_samples", use_container_width=True):
            if not sel_samples:
                st.error(f"Select at least 1 and at most {MAX_SAMPLE_SELECTION} sample files.")
            else:
                st.session_state.pending_action = "samples"
                st.session_state.pending_files = sel_samples
                st.session_state.processing = True
                st.session_state.page = "Processing"
                st.rerun()

    st.write("")
    st.markdown(f"### ⬆️ Upload WAV files (max {MAX_UPLOAD_FILES})")
    uploaded_files = st.file_uploader(
        "Drop .wav files here",
        type=["wav"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.get('uploader_key', 0)}"
    )

    if uploaded_files and len(uploaded_files) > MAX_UPLOAD_FILES:
        st.error(f"You can upload at most {MAX_UPLOAD_FILES} files at a time.")

    upload_col1, upload_col2 = st.columns([3, 1])
    with upload_col2:
        if st.button("Proceed Upload", key="proceed_upload", use_container_width=True):
            if not uploaded_files:
                st.error(f"Please upload at least 1 and at most {MAX_UPLOAD_FILES} .wav files.")
            else:
                uploaded_names = []
                for f in uploaded_files[:MAX_UPLOAD_FILES]:
                    name = f.name
                    if not name.lower().endswith(".wav"):
                        continue
                    data = f.read()
                    upload_blob(name, data)
                    uploaded_names.append(name)
                st.session_state.pending_action = "upload"
                st.session_state.pending_files = uploaded_names
                st.session_state.processing = True
                st.session_state.page = "Processing"
                st.rerun()

    # Optionally show confirmation modal if pending action is set
    if st.session_state.get("pending_action") in ("samples", "upload") and st.session_state.get("pending_files"):
        confirmation_modal()
